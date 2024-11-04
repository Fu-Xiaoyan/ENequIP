from typing import Dict, Union
import torch
from .utils_torch import radius_graph, scatter, scatter_mean, scatter_std
from e3nn import o3
from .message_passing_torch import MessagePassing
from .field_layer import FieldLayer
from .embedding_torch import *
from e3nn.o3 import TensorProduct
from .convolution_torch import shift_softplus, FullyConnectedTensorProduct, Linear
from e3nn.nn import NormActivation

class AtomwiseLinear(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_output = o3.Irreps(irreps_out)
        self.linear = o3.Linear(self.irreps_in, self.irreps_output)
        
    def forward(self, node_input):
        return self.linear(node_input)
    
    def __repr__(self):
        return self.linear.__repr__()


class EnergyNet(torch.nn.Module):

    def __init__(
        self,
        irreps_embedding_out,
        irreps_conv_out,
        r_max,
        num_layers,
        num_basis,
        cutoff_p,
        hidden_mul,
        lmax,
        convolution_kwargs,
        databased_setting,
        field=False,
    ):
        super().__init__()

        self.r_max = r_max
        self.irreps_conv_out = o3.Irreps(irreps_conv_out)

        self.irreps_embedding_out = o3.Irreps(irreps_embedding_out)
        element_number = databased_setting['element_number']
        atomic_scale = databased_setting['atomic_scale']
        atomic_shift = databased_setting['atomic_shift']

        irreps_node_hidden = o3.Irreps([(hidden_mul, (l, p))
                                    for l in range(lmax + 1) for p in [-1, 1]])
        
        self.one_hot = OneHot(element_number)
        self.sh = o3.SphericalHarmonics(range(lmax+1), True, normalization="component")
        self.radial_embedding = RadialEdgeEmbedding(r_max, num_basis, cutoff_p)

        self.lin_input = AtomwiseLinear(self.one_hot.irreps_output, irreps_conv_out)    # chemical embedding

        irreps_edge_scalars = self.radial_embedding.irreps_out

        self.mp = MessagePassing(
            irreps_node_input=self.lin_input.irreps_output,
            irreps_node_attr=self.one_hot.irreps_output,
            irreps_node_hidden=irreps_node_hidden,
            irreps_node_output=self.irreps_conv_out,
            irreps_edge_attr=self.sh.irreps_out,
            irreps_edge_scalars=irreps_edge_scalars,
            num_layers=num_layers,
            resnet=True,
            convolution_kwargs=convolution_kwargs,
        )
        # E_field
        if field:
            self.etp = FieldLayer(
                irreps_node_input=self.mp._irreps_out,
                irreps_node_attr=self.one_hot.irreps_output,
                irreps_node_output=self.mp._irreps_out,
                resnet=False
            )
        else:
            self.etp = None

        self.lin1 = o3.Linear(
            self.mp._irreps_out,
            self.irreps_embedding_out
        )
        irreps_out = '1x0e'

        self.lin2 = o3.Linear(
            self.irreps_embedding_out, 
            irreps_out,
        )

        if atomic_scale:
            atomic_scale = torch.as_tensor(atomic_scale)
            self.register_buffer("atomic_scale", atomic_scale)
        if atomic_shift:
            atomic_shift = torch.as_tensor(atomic_shift)
            self.register_buffer("atomic_shift", atomic_shift)

    def forward(self, atom_type, atom_pos, edge_src, edge_dst, edge_cell_shift, cell, image_index, field=None):
        edge_vec = atom_pos[edge_dst] - atom_pos[edge_src]
        cell = cell.view(-1, 3, 3)
        edge_vec = edge_vec + torch.einsum("ni,nij->nj", edge_cell_shift, cell[image_index[edge_src]])
        node_inputs = self.one_hot(atom_type)
        node_attr = node_inputs.copy_(node_inputs)
        edge_attr = self.sh(edge_vec)

        edge_length = edge_vec.norm(None, 1)
        edge_length_embedding = self.radial_embedding(edge_length)

        node_features = self.lin_input(node_inputs)
        node_features = self.mp(node_features, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)

        if self.etp:
            node_features = self.etp(node_features, node_attr, field)

        node_features = self.lin1(node_features)
        atomic_energy = self.lin2(node_features)

        if hasattr(self, 'atomic_scale') and self.atomic_scale:
            atomic_energy = atomic_energy * self.atomic_scale
        if hasattr(self, 'atomic_shift') and self.atomic_shift:
            atomic_energy = atomic_energy + self.atomic_shift
        return atomic_energy


class ForceNet(torch.nn.Module):
    def __init__(
            self,
            irreps_embedding_out='16x0e',
            irreps_conv_out='32x0e',
            r_max=4.0,
            num_layers=3,
            num_basis=8,
            cutoff_p=6,
            hidden_mul=50,
            lmax=1,
            convolution_kwargs={'invariant_layers':2, 'invariant_neurons':64, 'avg_num_neighbors':20, 'use_sc':True},
            databased_setting={'element_number': 2, 'atomic_scale': 1, 'atomic_shift': 0},
            field=False,
            is_training=True
    ):
        super().__init__()
        self.r_max = r_max
        self.EnergyNet = EnergyNet(irreps_embedding_out,
                                   irreps_conv_out,
                                   r_max, num_layers,
                                   num_basis,
                                   cutoff_p,
                                   hidden_mul,
                                   lmax,
                                   convolution_kwargs,
                                   databased_setting,
                                   field
                                   )
        self.training = is_training

    def forward(self, atom_type, atom_pos, edge_src, edge_dst, edge_cell_shift, cell, image_index, field=None):
        # in case prediction
        original_requires_grad = atom_pos.requires_grad
        atom_pos.requires_grad_(True)  # for force prediction, grad is necessary
        atomic_energy = self.EnergyNet(atom_type, atom_pos, edge_src, edge_dst, edge_cell_shift, cell, image_index, field)

        grads = torch.autograd.grad(
            [atomic_energy.sum()],  # sum all atomic energy batchwise
            atom_pos,
            create_graph=self.training,  # needed to allow gradients of this output during training
        )
        forces = torch.neg(grads[0])
        atom_pos.requires_grad_(original_requires_grad)

        return atomic_energy, forces


class ElePotentialNet(torch.nn.Module):
    def __init__(
            self,
            irreps_embedding_out='16x0e',
            irreps_conv_out='32x0e',
            r_max=4.0,
            num_layers=3,
            num_basis=8,
            cutoff_p=6,
            hidden_mul=50,
            lmax=1,
            convolution_kwargs={'invariant_layers':2, 'invariant_neurons':64, 'avg_num_neighbors':20, 'use_sc':True},
            databased_setting={'element_number': 2, 'atomic_scale': 1, 'atomic_shift': 0},
            field=False,
            is_training=True
    ):
        super().__init__()
        self.r_max = r_max
        self.irreps_conv_out = o3.Irreps(irreps_conv_out)
        self.irreps_embedding_out = o3.Irreps(irreps_embedding_out)
        self.r_max = r_max
        self.irreps_conv_out = o3.Irreps(irreps_conv_out)

        self.irreps_embedding_out = o3.Irreps(irreps_embedding_out)
        element_number = databased_setting['element_number']
        irreps_node_hidden = o3.Irreps([(hidden_mul, (l, p))
                                        for l in range(lmax + 1) for p in [-1, 1]])

        self.one_hot = OneHot(element_number)
        self.sh = o3.SphericalHarmonics(range(lmax + 1), True, normalization="component")
        self.radial_embedding = RadialEdgeEmbedding(r_max, num_basis, cutoff_p)

        self.lin_input = AtomwiseLinear(self.one_hot.irreps_output, irreps_conv_out)  # chemical embedding

        irreps_edge_scalars = self.radial_embedding.irreps_out

        self.mp = MessagePassing(
            irreps_node_input=self.lin_input.irreps_output,
            irreps_node_attr=self.one_hot.irreps_output,
            irreps_node_hidden=irreps_node_hidden,
            irreps_node_output=self.irreps_conv_out,
            irreps_edge_attr=self.sh.irreps_out,
            irreps_edge_scalars=irreps_edge_scalars,
            num_layers=num_layers,
            resnet=True,
            convolution_kwargs=convolution_kwargs,
        )
        irreps_out = '1x0e'
        self.vaccum_lin1 = o3.Linear(
            self.mp._irreps_out,
            self.irreps_embedding_out
        )

        self.vaccum_lin2 = o3.Linear(
            self.irreps_embedding_out,
            irreps_out,
        )
        self.fermi_lin1 = o3.Linear(
            self.mp._irreps_out,
            self.irreps_embedding_out
        )

        self.fermi_lin2 = o3.Linear(
            self.irreps_embedding_out,
            irreps_out,
        )

    def forward(self, atom_type, atom_pos, edge_src, edge_dst, edge_cell_shift, cell, image_index, field=None):
        edge_vec = atom_pos[edge_dst] - atom_pos[edge_src]
        cell = cell.view(-1, 3, 3)
        edge_vec = edge_vec + torch.einsum("ni,nij->nj", edge_cell_shift, cell[image_index[edge_src]])
        node_inputs = self.one_hot(atom_type)
        node_attr = node_inputs.copy_(node_inputs)
        edge_attr = self.sh(edge_vec)

        edge_length = edge_vec.norm(None, 1)
        edge_length_embedding = self.radial_embedding(edge_length)

        node_features = self.lin_input(node_inputs)
        node_features = self.mp(node_features, node_attr, edge_src, edge_dst, edge_attr, edge_length_embedding)

        # add direction tp  norm of the slab

        vaccum_feature = scatter_mean(node_features, image_index, dim=0)
        vaccum_feature = self.vaccum_lin1(vaccum_feature)
        vaccum = self.vaccum_lin2(vaccum_feature)

        fermi_feature = scatter_std(node_features, image_index, dim=0)
        fermi_feature = self.fermi_lin1(fermi_feature)
        fermi = self.fermi_lin2(fermi_feature)
        return fermi, vaccum
