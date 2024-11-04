import torch
import numpy as np
from dataset import DataLoader, ase_dataset_reader
from nn.network_torch import EnergyNet, ForceNet
from nn.utils_torch import scatter
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
import json


def train_loop(dataset, net, loss_fn, optimizer, force_weight=1.0, batch_size=5):
    total_loss = 0
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, **dl_kwargs)
    size = len(dataset)
    for ibatch, batch_data in enumerate(dataloader):

        atom_type, atom_pos, edge_src, edge_dst, edge_cell_shift, cell, image_index, \
        image_energy, atomic_forces, field = batch_data
        atomic_energy, pre_forces = net(atom_type, atom_pos, edge_src, edge_dst, edge_cell_shift, cell, image_index, field)

        pred = scatter(atomic_energy, image_index, dim=0)
        image_energy = image_energy.unsqueeze(-1)

        energy_loss = loss_fn(pred, image_energy)  # per image loss
        force_loss = loss_fn(pre_forces, atomic_forces)  # per atom loss
        loss = energy_loss * batch_size / len(atom_type) + force_weight * force_loss   # per atom loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ibatch % 10 == 0:
            loss_, current = loss.item(), (ibatch + 1) * batch_size
            print(f" {current / size * 100:5.2f}%    loss: {loss:>7f}")
        total_loss += loss.item()
    return total_loss / size * batch_size


ase_kwargs = {'filename': 'CuOwithErefined.extxyz', 'format': 'extxyz'}
checkpoint_name = 'CuOwithErefined.pth'
atomicdata_kwargs = {'r_max': 4.0}
device = 'cuda:0'
dataset = ase_dataset_reader(ase_kwargs, atomicdata_kwargs, device=device)
# if E_field, field should save as 'initial_magmoms' in extxyz
dl_kwargs = dict(atomic_map=['Cu', 'O'], device=device, E_field=True)
net_config = dict(
    irreps_embedding_out='16x0e',
    irreps_conv_out='32x0e',
    r_max=atomicdata_kwargs['r_max'],
    num_layers=4,
    num_basis=8,
    cutoff_p=6,
    hidden_mul=32,
    lmax=1,
    convolution_kwargs={'invariant_layers': 2, 'invariant_neurons': 64, 'avg_num_neighbors': 20, 'use_sc': True},
    databased_setting={'element_number': len(dl_kwargs['atomic_map']), 'atomic_scale': 2.27, 'atomic_shift': -3.63},
    field=True,
)
net = ForceNet(**net_config)
config = (atomicdata_kwargs, device, dl_kwargs, net_config)
json.dump(config, open('config.json', 'w'))
net.cuda()
# net.load_state_dict(torch.load(checkpoint_name))
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)
scheduler = ReduceLROnPlateau(optimizer, patience=100, min_lr=1e-5, factor=0.5, verbose=True)
epoch_size = 100
loss_his = []
for epoch in range(epoch_size):
    print(f"Epoch {epoch + 1}: ")
    loss_ = train_loop(dataset, net, loss_fn, optimizer, force_weight=1.0, batch_size=20)
    scheduler.step(metrics=loss_)
    if epoch % 10 == 0:
        torch.save(net.state_dict(), checkpoint_name)
    loss_his += [loss_]

torch.save(net.state_dict(), checkpoint_name)
import matplotlib.pyplot as plt
plt.semilogy(np.arange(1, epoch_size+1), loss_his)
plt.xlabel('epochs')
plt.ylabel('total_loss MSE/image')
plt.ylim([0, 1])
print(loss_his)
plt.show()
