# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import math

import numpy as np

import torch
from torch import nn

from e3nn import o3


@torch.jit.script
def _poly_cutoff(x: torch.Tensor, factor: float, p: float = 6.0) -> torch.Tensor:
    x = x * factor

    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))

    return out * (x < 1.0)


class PolyCutoff(torch.nn.Module):
    _factor: float
    p: float

    def __init__(self, r_max: float, p: float = 6):
        r"""Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        Parameters
        ----------
        r_max : float
            Cutoff radius
        p : int
            Power used in envelope function
        """
        super().__init__()
        assert p >= 2.0
        self.p = float(p)
        self._factor = 1.0 / float(r_max)

    def forward(self, x):
        """
        Evaluate cutoff function.
        x: torch.Tensor, input distance
        """
        return _poly_cutoff(x, self._factor, p=self.p)
    

class BesselBasis(nn.Module):
    r_max: float
    prefactor: float

    def __init__(self, r_max, num_basis=8, trainable=True):
        r"""Radial Bessel Basis, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        Parameters
        ----------
        r_max : float
            Cutoff radius
        num_basis : int
            Number of Bessel Basis functions
        trainable : bool
            Train the :math:`n \pi` part or not.
        """
        super(BesselBasis, self).__init__()

        self.trainable = trainable
        self.num_basis = num_basis

        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max

        bessel_weights = torch.linspace(start=1.0, end=num_basis, steps=num_basis) * math.pi
        if self.trainable:
            self.bessel_weights = nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Bessel Basis for input x.
        Parameters
        ----------
        x : torch.Tensor
            Input
        """
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)

        return self.prefactor * (numerator / x.unsqueeze(-1))
    

class RadialEdgeEmbedding(nn.Module):
    def __init__(self, r_max, num_basis=8, p=6):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff_p = p
        self.basis = BesselBasis(r_max, num_basis)
        self.cutoff = PolyCutoff(r_max, p)

        self.irreps_out = o3.Irreps([(self.basis.num_basis, (0, 1))])

    def forward(self, edge_length):
        edge_length_embedded = self.basis(edge_length) * self.cutoff(edge_length).unsqueeze(-1)
        return edge_length_embedded
        
    def __repr__(self):
        return f'RadialEdgeEmbedding [num_basis: {self.num_basis}, cutoff_p: {self.cutoff_p}] ( -> {self.irreps_out} | {self.basis.num_basis} weights)'
    

class OneHot(nn.Module):

    def __init__(self, num_types):
        super().__init__()
        self.num_types = num_types
        self.irreps_output = o3.Irreps([(self.num_types, (0, 1))])

    def forward(self, atom_type):
        type_numbers = atom_type
        one_hot = torch.nn.functional.one_hot(type_numbers, self.num_types)
        return one_hot
    
    def __repr__(self):
        return f'OneHot [num_types: {self.num_types}] ( -> {self.irreps_output})'