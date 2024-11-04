import torch
import numpy as np
from dataset import DataLoader, ase_dataset_reader
from nn.network_torch import ElePotentialNet, ForceNet
from nn.utils_torch import scatter
from sklearn.metrics import mean_squared_error
import json
ase_kwargs = {'filename': 'EP_refineddata.extxyz', 'format': 'extxyz'}
atomicdata_kwargs, device, dl_kwargs, net_config = json.load(open('Upzc_config.json', 'r'))
dataset = ase_dataset_reader(ase_kwargs, atomicdata_kwargs, device=device)
net = ElePotentialNet(**net_config)
net.cuda()
net.load_state_dict(torch.load('EP_CuO.pth'))
net.eval()
dataloader = DataLoader(dataset, shuffle=True, batch_size=1, **dl_kwargs)
size = len(dataset)
DFT_fermi = []
NN_fermi = []

DFT_vaccum = []
NN_vaccum = []
for ibatch, batch_data in enumerate(dataloader):
    atom_type, atom_pos, edge_src, edge_dst, edge_cell_shift, cell, image_index, \
    fermi, vaccum, field = batch_data
    pred_fermi, pred_vaccum = net(atom_type, atom_pos, edge_src, edge_dst, edge_cell_shift, cell, image_index, field)
    DFT_fermi.extend(list(torch.reshape(fermi, (-1,)).cpu().numpy()))
    NN_fermi.extend(list(torch.reshape(pred_fermi, (-1,)).detach().cpu().numpy()))
    DFT_vaccum.extend(list(torch.reshape(vaccum, (-1,)).cpu().numpy()))
    NN_vaccum.extend(list(torch.reshape(pred_vaccum, (-1,)).detach().cpu().numpy()))


import matplotlib.pyplot as plt

plt.scatter(DFT_fermi, NN_fermi)
plt.xlabel('DFT')
plt.ylabel('ML')
min_ = min(min(NN_fermi), min(DFT_fermi))
max_ = max(max(NN_fermi), max(DFT_fermi))
plt.plot([min_, max_], [min_, max_])
plt.title('Fermi level (eV)')
RMSE = np.sqrt(mean_squared_error(DFT_fermi, NN_fermi))
plt.text(max_ - (max_ - min_) * 0.25, min_, f'RMSE = {RMSE:.3f}')

plt.figure()
plt.scatter(DFT_vaccum, NN_vaccum)
plt.xlabel('DFT')
plt.ylabel('ML')
min_ = min(min(NN_vaccum), min(DFT_vaccum))
max_ = max(max(NN_vaccum), max(DFT_vaccum))
plt.plot([min_, max_], [min_, max_])
plt.title('Vaccum level (V)')
RMSE = np.sqrt(mean_squared_error(DFT_vaccum, NN_vaccum))
plt.text(max_ - (max_ - min_) * 0.25, min_, f'RMSE = {RMSE:.3f}')

DFT_Upzc = [-phi - Ef for phi, Ef in zip(DFT_vaccum, DFT_fermi)]
NN_Upzc = [-phi - Ef for phi, Ef in zip(NN_vaccum, NN_fermi)]
plt.figure()
plt.scatter(DFT_Upzc, NN_Upzc)
plt.xlabel('DFT')
plt.ylabel('ML')
min_ = min(min(NN_Upzc), min(DFT_Upzc))
max_ = max(max(NN_Upzc), max(DFT_Upzc))
plt.plot([min_, max_], [min_, max_])
plt.title('WF (eV)')
RMSE = np.sqrt(mean_squared_error(DFT_Upzc, NN_Upzc))
plt.text(max_ - (max_ - min_) * 0.25, min_, f'RMSE = {RMSE:.3f}')

plt.show()