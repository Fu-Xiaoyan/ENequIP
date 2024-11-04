from typing import List
from .torch_geometric import Batch, Data
import random
import numpy as np
from ase.symbols import Symbols, symbols2numbers
from torch import tensor as Tensor
import torch
int_type = torch.int64
float_type = torch.float32
bool_type = torch.bool

class Collater(object):
    """Collate a list of ``AtomicData``.

    Args:
        fixed_fields: which fields are fixed fields
        exclude_keys: keys to ignore in the input, not copying to the output
    """

    def __init__(
        self,
        fixed_fields: List[str] = [],
        exclude_keys: List[str] = [],
    ):
        self.fixed_fields = fixed_fields
        self._exclude_keys = set(exclude_keys)

    @classmethod
    def for_dataset(
        cls,
        dataset,
        exclude_keys: List[str] = [],
    ):
        """Construct a collater appropriate to ``dataset``.

        All kwargs besides ``fixed_fields`` are passed through to the constructor.
        """
        return cls(
            fixed_fields=list(getattr(dataset, "fixed_fields", {}).keys()),
            exclude_keys=exclude_keys,
        )

    def collate(self, batch: List[Data]) -> Batch:
        """Collate a list of data"""
        # For fixed fields, we need to batch those that are per-node or
        # per-edge, since they need to be repeated in order to have the same
        # number of nodes/edges as the full batch graph.
        # For fixed fields that are per-example, however — those with __cat_dim__
        # of None — we can just put one copy over the whole batch graph.
        # Figure out which ones those are:
        new_dim_fixed = set()
        for f in self.fixed_fields:
            if batch[0].__cat_dim__(f, None) is None:
                new_dim_fixed.add(f)
        # TODO: cache ^ and the batched versions of fixed fields for various batch sizes if necessary for performance
        out = Batch.from_data_list(
            batch, exclude_keys=self._exclude_keys.union(new_dim_fixed)
        )
        for f in new_dim_fixed:
            if f in self._exclude_keys:
                continue
            out[f] = batch[0][f]
        return out

    def __call__(self, batch: List[Data]) -> Batch:
        """Collate a list of data"""
        return self.collate(batch)

    @property
    def exclude_keys(self):
        return list(self._exclude_keys)

# a rewrite data loader


class DataLoader(object):
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        exclude_keys: List[str] = [],
        image_detail = False,
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]
        self.batch_size = batch_size
        self.atomic_map = symbols2numbers(kwargs['atomic_map'])
        self.index = 0
        self.collate_fn = Collater.for_dataset(dataset, exclude_keys=exclude_keys)
        self.dataset = dataset
        self.device = kwargs.get('device')
        self.E_field = kwargs.get('E_field', False)
        self.ElePotential = kwargs.get('ElePotential', False)

        # drop_last
        dlsize = len(self.dataset) // self.batch_size * self.batch_size
        # random.seed(0)
        self.indexlist = [i for i in range(dlsize)]
        if shuffle:
            random.shuffle(self.indexlist)
        else:
            self.indexlist = [i for i in range(dlsize)]
        self.sampler = iter(np.array(self.indexlist).reshape(-1, batch_size))

    def __iter__(self):
        return self

    def __next__(self):
        index = next(self.sampler)
        input_data = [self.dataset[idx] for idx in index]
        input_data = self.collate_fn(input_data)
        atom_type = self.trans_atomic(input_data.atomic_numbers.squeeze())
        atom_pos = input_data.pos
        edge_src = input_data.edge_index[0]
        edge_dst = input_data.edge_index[1]
        edge_cell_shift = input_data.edge_cell_shift
        cell = input_data.cell
        image_index = input_data.batch
        image_energy = input_data.free_energy
        atomic_forces = input_data.forces
        if self.E_field:
            field = input_data.E_field
        else:
            field = None
        if self.ElePotential:
            fermi = input_data.fermi
            vaccum = input_data.vaccum
            batch_data = atom_type, atom_pos, edge_src, edge_dst, edge_cell_shift, cell, image_index, \
                         fermi, vaccum, field
            return batch_data
        batch_data = atom_type, atom_pos, edge_src, edge_dst, edge_cell_shift, cell, image_index, \
                     image_energy, atomic_forces, field
        return batch_data

    def __len__(self):
        return len(self.dataset)

    def trans_atomic(self, atomic_number):
        atom_type = []
        for atom_index in atomic_number:
            atom_type.append(self.atomic_map.index(atom_index))
        atom_type = Tensor(atom_type, dtype=int_type, device=self.device)
        return atom_type

