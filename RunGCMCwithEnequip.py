from ASEcalculator import EnequiPcalculator
import torch
import ase.io
from optimizer import GCMC
from ase.build import molecule
import os

def main(UvsRHE, index, save_dir):
    # NN config
    net_config = dict(
        irreps_embedding_out='16x0e',
        irreps_conv_out='32x0e',
        r_max=4.0,
        num_layers=4,
        num_basis=8,
        cutoff_p=6,
        hidden_mul=32,
        lmax=1,
        convolution_kwargs={'invariant_layers': 2, 'invariant_neurons': 64, 'avg_num_neighbors': 20, 'use_sc': True},
        databased_setting={'element_number': 2, 'atomic_scale': 2.27, 'atomic_shift': -3.63},
        field=True,
        device='cuda:0',
        atomic_map=['Cu', 'O']
    )
    calculator = EnequiPcalculator.load_model(net_config, checkpoint=torch.load('CuO-0615.pth'))

    # Atoms preprocess
    atoms = ase.io.read('gcmc-results-backup/Cu2O_111.traj')
    bulk = "Cu2O"
    save_name = "Cu2O111refH2O_"
    # GCMC optimizer
    # UvsRHE = -0.1
    ads = [molecule("O")]
    u_ref = [-7.087 + UvsRHE * 2]

    for atom in atoms:
        atom.magmom = [0, 0, 0]   # for calculator initialize
        if atom.symbol == 'O' and atom.position[2] > 11:
            atom.tag = 1   # tag oxygen for MC removing
        else:
            atom.tag = 0

    potential_config = dict(
        URHE=UvsRHE,
        pH=1,
        capacitance=20,   # uF/cm-2
        dielectric_constant=3,  # dielectric constant of water near a surface is 2 (DOI: 10.1126/science.aat4191)
        vaccum_permittivity=8.85E-12,      # F/m
        USHE=4.4,
        Upzc_net='EP_CuO0623.pth',
        Upzc_net_config='Upzc_config.json',
    )

    atoms.set_calculator(calculator)
    dyn = GCMC(atoms=atoms,
               ads=ads,
               calc=calculator,
               u_ref=u_ref,
               logfile=save_dir + '/' + save_name+str(UvsRHE)+'_'+str(index)+'.log',
               trajectory=save_dir + '/' + save_name+str(UvsRHE)+'_'+str(index)+'.traj',
               temperature_K=298,
               seed=None,
               opt_steps=50,    # structure relaxation steps within each MC step
               opt_fmax=0.2,
               rmin=1.6,
               add=True,
               remove=True,
               displace=True,
               save_all=True,
               save_opt=False,
               steps=200,   # ????
               hight=None,
               magmom=None,
               potential_config=potential_config,
               only_surface=True, bulk=bulk,  # only remove surface atoms
               )
    dyn.run(steps=400)


if __name__ == '__main__':
    save_dir = "gcmcoutput"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    UvsRHEs=[0.25, 0, -0.25, -0.5, -0.75]
    for UvsRHE in UvsRHEs:
        for index in range(20):
            main(UvsRHE, index, save_dir)
