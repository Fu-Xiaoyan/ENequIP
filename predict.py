
import torch
import numpy as np
from dataset import DataLoader, ase_dataset_reader
from nn.network_torch import EnergyNet, ForceNet
from nn.utils_torch import scatter
from sklearn.metrics import mean_squared_error
import json
ase_kwargs = {'filename': 'CuOwithErefinedAimd.extxyz', 'format': 'extxyz'}
atomicdata_kwargs, device, dl_kwargs, net_config = json.load(open('config.json', 'r'))
dataset = ase_dataset_reader(ase_kwargs, atomicdata_kwargs, device=device)
net = ForceNet(**net_config)
net.cuda()
net.load_state_dict(torch.load('CuOwithErefined.pth'))
net.eval()
dataloader = DataLoader(dataset, shuffle=True, batch_size=1, **dl_kwargs)
size = len(dataset)
DFT_e = []
NN_e = []

DFT_f = []
NN_f = []
for ibatch, batch_data in enumerate(dataloader):
    atom_type, atom_pos, edge_src, edge_dst, edge_cell_shift, cell, image_index, \
    image_energy, atomic_forces, field = batch_data
    atomic_energy, pre_forces = net(atom_type, atom_pos, edge_src, edge_dst, edge_cell_shift, cell, image_index, field)
    pred_e = scatter(atomic_energy, image_index, dim=0)

    DFT_e.extend(list(torch.reshape(image_energy, (-1,)).cpu().numpy()))
    NN_e.extend(list(torch.reshape(pred_e, (-1,)).detach().cpu().numpy()))
    DFT_f.extend(list(torch.reshape(atomic_forces, (-1,)).cpu().numpy()))
    NN_f.extend(list(torch.reshape(pre_forces, (-1,)).cpu().numpy()))


import matplotlib.pyplot as plt

plt.scatter(DFT_e, NN_e)
plt.xlabel('DFT')
plt.ylabel('ML')
min_ = min(min(NN_e), min(DFT_e))
max_ = max(max(NN_e), max(DFT_e))
plt.plot([min_, max_], [min_, max_])
plt.title('Energy (eV)')
RMSE = np.sqrt(mean_squared_error(DFT_e, NN_e))
plt.text(max_ - (max_ - min_) * 0.25, min_, f'RMSE = {RMSE:.3f}')

# forces
plt.figure()
plt.scatter(DFT_f, NN_f)
plt.xlabel('DFT')
plt.ylabel('ML')
min_ = min(min(NN_f), min(DFT_f))
max_ = max(max(NN_f), max(DFT_f))
plt.plot([min_, max_], [min_, max_])
plt.title('Forces (eV/A)')
RMSE = np.sqrt(mean_squared_error(DFT_f, NN_f))
plt.text(max_ - (max_ - min_) * 0.25, min_, f'RMSE = {RMSE:.3f}')

plt.show()