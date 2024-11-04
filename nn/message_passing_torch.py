
import torch
from e3nn import o3
from e3nn.nn import Gate, NormActivation

from .convolution_torch import Convolution, shift_softplus


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *input):
        x = self.first(*input)
        x = self.second(x)
        return x


acts = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "ssp": shift_softplus,
    "silu": torch.nn.functional.silu,
}


class MessagePassing(torch.nn.Module):
    r"""
    Parameters
    ----------
    irreps_node_input : `e3nn.o3.Irreps`
        representation of the input features
    irreps_node_hidden : `e3nn.o3.Irreps`
        representation of the hidden features
    irreps_node_output : `e3nn.o3.Irreps`
        representation of the output features
    irreps_node_attr : `e3nn.o3.Irreps`
        representation of the nodes attributes
    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes
    layers : int
        number of gates (non linearities)
    fc_neurons : list of int
        number of neurons per layers in the fully connected network
        first layer and hidden layers but not the output layer
    """

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_node_hidden,
        irreps_node_output,
        irreps_edge_attr,
        irreps_edge_scalars,
        convolution_kwargs={},
        num_layers=3,
        resnet=False,
        nonlin_type="gate",
        nonlin_scalars={"e": "ssp", "o": "tanh"},
        nonlin_gates={"e": "ssp", "o": "abs"}
    ) -> None:
        super().__init__()
        if not nonlin_type in ("gate", "norm"):
            raise ValueError(f"Unexpected nonlin_type {nonlin_type}.")

        nonlin_scalars = {
            1: nonlin_scalars["e"],
            -1: nonlin_scalars["o"],
        }
        nonlin_gates = {
            1: nonlin_gates["e"],
            -1: nonlin_gates["o"],
        }

        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_hidden = o3.Irreps(irreps_node_hidden)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_edge_scalars = o3.Irreps(irreps_edge_scalars)

        irreps_node = self.irreps_node_input
        irreps_prev = irreps_node
        self.layers = torch.nn.ModuleList()
        self.resnets = []

        self.layers = torch.nn.ModuleList()

        for _ in range(num_layers):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_node_hidden
                    if ir.l == 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
                ]
            ).simplify()
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_node_hidden
                    if ir.l > 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
                ]
            )
            if nonlin_type == "gate":
                ir = "0e" if tp_path_exists(irreps_node, self.irreps_edge_attr, "0e") else "0o"
                irreps_gates = o3.Irreps([(mul, ir)
                                      for mul, _ in irreps_gated]).simplify()

                nonlinear = Gate(
                    irreps_scalars,
                    [acts[nonlin_scalars[ir.p]] for _, ir in irreps_scalars],
                    irreps_gates,
                    [acts[nonlin_gates[ir.p]] for _, ir in irreps_gates],
                    irreps_gated,
                )

                conv_irreps_out = nonlinear.irreps_in.simplify()
            else:
                conv_irreps_out = (irreps_scalars + irreps_gated).simplify()

                nonlinear = NormActivation(
                    irreps_in=conv_irreps_out,
                    act=acts[nonlin_scalars[1]],
                    normalize=True,
                    epsilon=1e-8,
                    bias=False,
                )
            conv = Convolution(
                irreps_node_input=irreps_node,
                irreps_node_attr=self.irreps_node_attr,
                irreps_node_output=conv_irreps_out,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_edge_scalars=self.irreps_edge_scalars,
                **convolution_kwargs,
            )
            irreps_node = nonlinear.irreps_out

            self.layers.append(Compose(conv, nonlinear))

            if irreps_prev == irreps_node and resnet:
                self.resnets.append(True)
            else:
                self.resnets.append(False)
            irreps_prev = irreps_node

        self._irreps_out = irreps_node
        self.resnets.append(False)

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        layer_in = node_input
        for i in range(len(self.layers)):
            layer_out = self.layers[i](
                layer_in, node_attr, edge_src, edge_dst, edge_attr, edge_scalars)

            if self.resnets[i]:
                layer_in = layer_out + layer_in
            else:
                layer_in = layer_out

        return layer_in

