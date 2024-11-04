import json
from ase.calculators.calculator import Calculator, all_changes
import torch
from nn.network_torch import EnergyNet, ForceNet
from dataset.asedata import AtomicData
import ase.io
from ase.symbols import symbols2numbers
from nn.network_torch import ElePotentialNet
from torch import tensor as Tensor
int_type = torch.int64
float_type = torch.float32
bool_type = torch.bool


class EnequiPcalculator(Calculator):
    """
    ASE Calculator.
    """
    implemented_properties = ["energy", "energies", "forces", "free_energy"]
    def __init__(
            self,
            net,
            **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self.model = net
        self.model.cuda()
        self.model.eval()

    @classmethod
    def load_model(
            cls,
            net_config,
            checkpoint,
            **kwargs
    ):
        cls.device = net_config['device']
        cls.atomic_map = symbols2numbers(net_config['atomic_map'])
        # load model
        net_config.pop('device')
        net_config.pop('atomic_map')
        net = ForceNet(**net_config)
        net.load_state_dict(checkpoint)
        return cls(net=net, **kwargs)

    def trans_atomic(self, atomic_number):
        atom_type = []
        for atom_index in atomic_number:
            atom_type.append(self.atomic_map.index(atom_index))
        atom_type = Tensor(atom_type, dtype=int_type, device=self.device)
        return atom_type

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
        """
        Calculate properties.

        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        atoms.wrap()
        Calculator.calculate(self, atoms)

        # prepare data
        input_data = AtomicData.from_ase(atoms=atoms, r_max=self.model.r_max)
        atom_type = self.trans_atomic(input_data.atomic_numbers.squeeze()).to(self.device)
        atom_pos = input_data.pos.to(self.device)
        edge_src = input_data.edge_index[0].to(self.device)
        edge_dst = input_data.edge_index[1].to(self.device)
        edge_cell_shift = input_data.edge_cell_shift.to(self.device)
        cell = input_data.cell.to(self.device)
        field = input_data.E_field.to(self.device)
        image_index = Tensor([0 for ll in range(len(atom_type))], dtype=int_type, device=self.device)
        atomic_energy, pre_forces = self.model(atom_type, atom_pos, edge_src, edge_dst, edge_cell_shift, cell, image_index,
                                        field)
        image_energy = atomic_energy.sum()

        self.results = {}
        # only store results the model actually computed to avoid KeyErrors
        self.results["energy"] = image_energy.detach().cpu().numpy()
        self.results["free_energy"] = self.results["energy"]
        self.results["energies"] = atomic_energy.detach().cpu().numpy()
        self.results["forces"] = pre_forces.detach().cpu().numpy()


def example():
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
    calculator = EnequiPcalculator.load_model(net_config, checkpoint=torch.load('CuOwithErefined.pth'))

    atoms = ase.io.read('XXXXXXXX.traj', index='0')
    E_apply = 0
    for atom in atoms:
        atom.magmom = [0, 0, E_apply]
    # calculator.calculate(atoms)
    atoms.set_calculator(calculator)
    E = atoms.get_potential_energy()
    F = atoms.get_forces()

    from ase.optimize import BFGS
    from ase.visualize import view
    from ase.constraints import FixAtoms
    c = FixAtoms(indices=[atom.index for atom in atoms if atom.position[2] < 12])
    atoms.set_constraint(c)

    dyn = BFGS(atoms, trajectory='XXXXXXXXXXXXX.traj')
    dyn.run(fmax=0.05,steps=10)

    view(ase.io.read('XXXXXXXXXXXX.traj', index=':'))


class Upzc_predictor(object):
    def __init__(self, config, ckpoint):
        atomicdata_kwargs, device, dl_kwargs, net_config = json.load(open(config, 'r'))
        self.net = ElePotentialNet(**net_config)
        self.net.cuda()
        self.net.load_state_dict(torch.load(ckpoint))
        self.net.eval()
        self.device = device
        self.atomic_map = symbols2numbers(dl_kwargs['atomic_map'])

    def trans_atomic(self, atomic_number):
        atom_type = []
        for atom_index in atomic_number:
            atom_type.append(self.atomic_map.index(atom_index))
        atom_type = Tensor(atom_type, dtype=int_type, device=self.device)
        return atom_type

    def get_Upzc(self, atoms):
        input_data = AtomicData.from_ase(atoms=atoms, r_max=self.net.r_max)
        atom_type = self.trans_atomic(input_data.atomic_numbers.squeeze()).to(self.device)
        atom_pos = input_data.pos.to(self.device)
        edge_src = input_data.edge_index[0].to(self.device)
        edge_dst = input_data.edge_index[1].to(self.device)
        edge_cell_shift = input_data.edge_cell_shift.to(self.device)
        cell = input_data.cell.to(self.device)
        field = None
        image_index = Tensor([0 for ll in range(len(atom_type))], dtype=int_type, device=self.device)
        pred_fermi, pred_vaccum = self.net(atom_type, atom_pos, edge_src, edge_dst, edge_cell_shift, cell, image_index, field)
        pred_fermi = pred_fermi.detach().cpu().numpy().item()
        pred_vaccum = pred_vaccum.detach().cpu().numpy().item()
        Upzc = pred_vaccum - pred_fermi
        return Upzc

def Upzc_example():
    atoms = ase.io.read('gcmcoutput/CuO_big.traj')
    Upzc_config = 'Upzc_config.json'
    ckpoint = 'EP_CuO.pth'
    pzc_predictor = Upzc_predictor(Upzc_config, ckpoint)
    Upzc = pzc_predictor.get_Upzc(atoms)


if __name__ == '__main__':
    example()
    # Upzc_example()


