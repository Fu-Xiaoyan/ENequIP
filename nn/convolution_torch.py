import math

import torch
import scipy

from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct, Linear
from e3nn.util.jit import compile_mode
from .utils_torch import scatter


@torch.jit.script
def shift_softplus(x):
    return torch.nn.functional.softplus(x) - math.log(2.0)

@compile_mode("script")
class Convolution(torch.nn.Module):
    r"""equivariant convolution
    Parameters
    ----------
    irreps_node_input : `e3nn.o3.Irreps`
        representation of the input node features
    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the node attributes
    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes
    irreps_node_output : `e3nn.o3.Irreps` or None
        representation of the output node features
    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer
    num_neighbors : float
        typical number of nodes convolved over
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_node_output,
        irreps_edge_attr,
        irreps_edge_scalars,
        invariant_layers=1,
        invariant_neurons=8,
        avg_num_neighbors=None,
        use_sc=True,
        nonlin_scalars={"e": "ssp"}
    ):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.use_sc = use_sc

        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_edge_scalars = o3.Irreps(
            [(irreps_edge_scalars.num_irreps, (0, 1))])

        self.lin1 = Linear(self.irreps_node_input, self.irreps_node_input)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [(i_1, i_2, p[i_out], mode, train) for i_1, i_2, i_out, mode, train in instructions]

        tp = TensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            [self.irreps_edge_scalars.num_irreps]
            + invariant_layers * [invariant_neurons]
            + [tp.weight_numel],
            {
                "ssp": shift_softplus,
                "silu": torch.nn.functional.silu,
            }[nonlin_scalars["e"]]
        )
        self.tp = tp

        self.lin2 = Linear(tp.irreps_out.simplify(), self.irreps_node_output)
        self.sc = None
        if self.use_sc:
            self.sc = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr,
                                    self.irreps_node_output)

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        weight = self.fc(edge_scalars)

        node_features = self.lin1(node_input)
        edge_features = self.tp(node_features[edge_dst], edge_attr, weight)  # mp trans from edge_dst to edge_src
        node_features = scatter(
            edge_features, edge_src, dim=0, dim_size=node_input.shape[0])

        if self.avg_num_neighbors is not None:
            node_features = node_features.div(self.avg_num_neighbors**0.5)

        node_features = self.lin2(node_features)

        if self.sc is not None:
            sc = self.sc(node_input, node_attr)
            node_features = node_features + sc

        return node_features
    
