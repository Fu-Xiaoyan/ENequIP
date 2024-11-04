import torch
from e3nn import o3
from e3nn.nn import NormActivation
from .convolution_torch import shift_softplus, FullyConnectedTensorProduct, Linear


class FieldLayer(torch.nn.Module):
    r"""
    Parameters
    ----------
    irreps_node_input : `e3nn.o3.Irreps`
        representation of the input features
    irreps_node_output : `e3nn.o3.Irreps`
        representation of the output features
    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the nodes attributes
    """

    def __init__(
            self,
            irreps_node_input,
            irreps_node_attr,
            irreps_node_output,
            resnet=True,
    ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_field_attr = o3.Irreps('1x1o')
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.resnets = resnet

        self.tp = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_field_attr,
                                              self.irreps_node_output, internal_weights=True)

        self.lin2 = Linear(self.tp.irreps_out.simplify(), self.irreps_node_output)
        self.sc = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr,
                                              self.irreps_node_output)

        self.nonlinear = NormActivation(
            irreps_in=self.irreps_node_output,
            scalar_nonlinearity=shift_softplus,
            normalize=True,
            epsilon=1e-8,
            bias=False,
        )

        self._irreps_out = self.nonlinear.irreps_out

    def forward(self, node_input, node_attr, field):
        node_features = self.tp(node_input, field)
        node_features = self.lin2(node_features)

        node_features = self.sc(node_features, node_attr)

        node_features = node_input + node_features

        node_features = self.nonlinear(node_features)
        return node_features

