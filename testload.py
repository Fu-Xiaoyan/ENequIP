# this script use to test the NN while coding.
# this script contains several function.
import torch
import numpy as np
from dataset import DataLoader, ase_dataset_reader
from nn.network_torch import EnergyNet, ForceNet
from nn.utils_torch import scatter
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
import json


def test_train_from_beginning():
    # test training from the very beginnig.
    pass



def test_dataset_loader():
    # test datasetloader, check the outputs.
    # check Qtot and atomic charges
    ase_kwargs = {'filename': 'all.extxyz', 'format': 'extxyz'}
    atomicdata_kwargs = {'r_max': 5.0}
    device = 'cuda:0'
    dataset = ase_dataset_reader(ase_kwargs, atomicdata_kwargs, device=device)
    dl_kwargs = dict(atomic_map=['Mn','Co', 'O'], device=device, charge=True)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=2, **dl_kwargs)
    for ibatch, batch_data in enumerate(dataloader):
        atom_type, atom_pos, edge_src, edge_dst, edge_cell_shift, cell, image_index, \
        image_energy, atomic_forces, field  = batch_data
        
test_dataset_loader()