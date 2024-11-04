import copy
import os
import shutil
import time
from math import ceil
import numpy as np
from numpy import sqrt
from numpy.random import rand
from ase import Atoms, data
from ase.io import write, Trajectory, read
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer, Dynamics
from ase.units import kB, pi, _hplanck, J, s
import math
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ASEcalculator import Upzc_predictor
from surfaceidentify.voronio_surface import find_surface_atoms_with_voronoi
from pymatgen.io.ase import AseAtomsAdaptor
from surfaceidentify.SurfaceAtomIdentifier import get_surface_atom_GCN

class GCMonteCarlo(Optimizer):

    defaults = {**Optimizer.defaults, 'dr': 0.25, "t": 298 * kB}

    def __init__(self, atoms, ads, calc,
                 u_ref=None, logfile='gcmc.log', label=None,
                 dr=None, temperature_K=None, displace_percentage=0.25, rmin=1.4, rmax=3.0,
                 optimizer=FIRE, trajectory=None, opt_fmax=None, opt_steps=None, seed=None,
                 save_opt=False, save_refused=False, save_all=False,
                 displace=True, remove=True, add=True, steps=0, magmom=None):
        """Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        ads: Atoms object
            The Atoms object need to be added on the atoms.

        calc: Calculator object
            In order to calculate forces and energies,
            you need to attach a calculator object to your atoms object

        u_ref : List
            The reference chemical potential of adsorbates.

        logfile: string
            Text file used to write summary information.

        dr: float
            Atomic displace step, default to 0.25.

        optimizer: ase Optimizer
            Default to use the FIRE to optimize the structure when accept trail.

        temperature_K: temperature
            Defaults to None. If users don't set value, the optimizer will use 298K

        trajectory: string
            Pickle file used to store trajectory of atomic movement.
            Attention: Due to the algorithm calls the SinglePointCalculator to storage the energy
            and the force of each step, the parameter don't need to be written
            (see gcmc.traj & all_trajectory.traj).

        seed: digital
            For a parallel calculation, it is important to use the same seed on all processors!

        save_opt: boolean
            Default to False. If set to True, the structure optimize trajectory will be restored.

        """
        restart = None
        master = None
        Optimizer.__init__(self, atoms, restart, logfile, master)

        global L , l, nano
        np.random.seed(seed=seed)

        self.traj = []
        self.sa = []
        self.sr = []
        self.all_traj = []
        self.tags_list = []
        self.save_all_step = []
        self.save_all = save_all
        self.atoms = atoms
        self.ads = self.transfer_ads(ads)
        self.calc = calc
        self.u_ref = u_ref
        self.optimizer = optimizer
        self.save_opt = save_opt
        self.dp = displace_percentage
        self.save_refused = save_refused
        self.rmin = rmin
        self.rmax = rmax
        structure = self.get_cluster_position(atoms)
        self.cluster = structure[0]
        self.nano = structure[1]
        self.displace = displace
        self.remove = remove
        self.add = add
        self.ad, self.u = self.get_ad_and_uref()
        self.magmom = magmom
        self.steps = steps

        if label is None:
            self.label = "default"
        elif label == "vasp":
            self.label = "vasp"
            self.vasp = copy.deepcopy(calc)
            self.vasp.set(nsw=steps)
        else:
            self.label = "default"

        if trajectory is None:
            self.Trajectory = "gcmc.traj"
        else:
            self.Trajectory = trajectory
        if dr is None:
            self.dr = self.defaults['dr']
        else:
            self.dr = dr
        if temperature_K is None:
            self.t = self.defaults['t']
            self.beta = 1 / self.t
            self.temperature = 298
        else:
            self.t = temperature_K * kB
            self.beta = 1 / self.t
            self.temperature = temperature_K
        if opt_fmax is None:
            self.opt_fmax = 0.1
        else:
            self.opt_fmax = opt_fmax
        if opt_steps is None:
            self.opt_steps = 100
        else:
            self.opt_steps = opt_steps

        if self.remove and not self.add:
            raise ValueError("In GCMC, remove particle must be associated with add particle. ")

    def initialize(self):
        self._dis = 0
        self._add = 0
        self._rem = 0
        self.trail1 = 0
        self.trail2 = 0
        self.trail3 = 0
        self.adsnum = 0
        self.tags = 111

    def irun(self, steps=None, **kwargs):
        """ call Dynamics.irun and keep track of fmax
        """
        self.fmax = 1e-10
        if steps:
            self.max_steps = steps
        return Dynamics.irun(self)

    def run(self, steps=None, **kwargs):
        """ call Dynamics.run and keep track of fmax"""
        self.fmax = 1e-10
        if steps:
            self.max_steps = steps
        return Dynamics.run(self)

    def transfer_ads(self, ads):
        images = []
        for ad in ads:
            image = Atoms()
            for atom in ad:
                image.append(atom)
            images.append(image)
        return images

    def compare_atoms(self, atoms1, atoms2):
        atoms1 = atoms1.copy()
        atoms2 = atoms2.copy()
        if atoms1 == atoms2:
            compare = True
        else:
            compare = False
        return compare

    def get_traj(self, traj, Traj=None):
        if Traj is None:
            Traj = []
        images = Trajectory(traj)
        for atoms in images:
            Traj.append(atoms)
        return Traj

    def get_adsnum(self, atoms):
        atoms = atoms.copy()
        tags = []
        for atom in atoms:
            if atom.tag >= 111:
                tags.append(atom.tag)
        tags = np.unique(tags)
        num = len(tags)
        return num

    def remove_dir(self, path):
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except OSError:
                pass
        else:
            pass

    def save_all_traj(self):
        if self.nsteps == self.max_steps - 1:
            if self.save_opt:
                write("all_opt.extxyz", self.all_traj)
                path = os.getcwd() + "/opt_traj/"
                self.remove_dir(path)
            if self.save_all:
                write("all_trial.extxyz", self.save_all_step)

    def structure_opt(self, atoms):
        if self.label == "default":
            if self.save_opt:
                path = os.getcwd() + "/opt_traj"
                self.create_file(path)
                traj_path = os.getcwd() + "/opt_traj/"
                dyn = self.optimizer(atoms, logfile=None,
                                     trajectory=traj_path + "step_%d_opt.traj" % (self.nsteps+1))
                dyn.run(fmax=self.opt_fmax, steps=self.opt_steps)
                traj = traj_path + "step_%d_opt.traj" % (self.nsteps+1)
                Traj = self.get_traj(traj)
                self.all_traj.extend(Traj)
                return atoms
            else:
                dyn = self.optimizer(atoms, logfile=None)
                dyn.run(fmax=self.opt_fmax, steps=self.opt_steps)
                return atoms
        else:
            if self.save_opt:
                path = os.getcwd() + "/opt_traj"
                self.create_file(path)
                traj_path = os.getcwd() + "/opt_traj/"
                atoms.calc = self.calc
                atoms.get_potential_energy()
                images = read("OUTCAR", index=":")
                write(traj_path + "step_%d_opt.traj" % (self.nsteps+1), images)
                self.all_traj.extend(images)
                return atoms
            else:
                atoms.calc = self.calc
                atoms.get_potential_energy()
                return atoms

    def create_file(self, filepath):
        if os.path.exists(filepath):
            pass
        else:
            try:
                os.mkdir(filepath)
            except OSError:
                pass

    def get_tag(self, atoms):

        tags = []
        atoms = atoms.copy()
        for atom in atoms:
            if atom.tag >= 111:
                tags.append(atom.tag)
        return tags

    def add_judgement(self, atoms):
        raise Exception('add_judgement for GCMC should be defined')

    def add_particle(self, ads):
        raise Exception('add_particle for GCMC should be defined')

    def save_accepted_atoms(self, atoms):
        image = atoms.copy()
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        image.calc = sp(atoms=atoms, energy=energy, forces=forces)
        self.sa.append(image)
        write(self.Trajectory, self.sa)

    def save_refused_atoms(self, atoms):
        image = atoms.copy()
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        image.calc = sp(atoms=atoms, energy=energy, forces=forces)
        self.sr.append(image)
        if self.nsteps == self.max_steps - 1:
            if self.save_refused:
                write("refused.traj", self.sr)

    def get_energy(self, atoms):
        if self.label == "default":
            calc = self.calc
            if self.steps:
                atoms.calc = calc
                dyn = self.optimizer(atoms, logfile=None)
                dyn.run(steps=self.steps)
            else:
                atoms.calc = calc

            energy = atoms.get_potential_energy()
            if self.save_all:
                image = atoms.copy()
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                image.calc = sp(atoms=atoms, energy=energy, forces=forces)
                self.save_all_step.append(image)
        else:
            calc = self.vasp
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            if self.save_all:
                image = read("OUTCAR")
                self.save_all_step.append(image)
        return energy

    def log(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces()
        fmax = sqrt((forces ** 2).sum(axis=1).max())
        e = self.atoms.get_potential_energy(
            force_consistent=self.force_consistent
        )
        if fmax > self.opt_fmax and self.nsteps == 0:
            self.atoms = self.structure_opt(self.atoms)
            forces = self.atoms.get_forces()
            fmax = sqrt((forces ** 2).sum(axis=1).max())
            e = self.atoms.get_potential_energy(
                force_consistent=self.force_consistent
            )
        T = time.localtime()
        if self.logfile is not None:
            trail = "initial"
            result = "initial"
            name = self.__class__.__name__
            if self.nsteps == 0:
                args = (" " * len(name), "Step", "Date", "Time", "Energy",
                        "fmax", "Temperature", "Trail", "Result",
                        "Adsorbates", "Numbers", "Uref", "E_applied",)
                msg = "%s  %4s %10s %8s %15s %12s %19s %12s %12s %15s %12s %10s %10s\n" % args
                self.logfile.write(msg)
                self.save_accepted_atoms(self.atoms)

                if self.force_consistent:
                    msg = "*Force-consistent energies used in optimization.\n"
                    self.logfile.write(msg)

            if self.trail1 == 1:
                if self._dis == 1:
                    trail = "displace"
                    result = "accepted"
                else:
                    trail = "displace"
                    result = "refused"
            if self.trail2 == 1:
                if self._rem == 1:
                    trail = "remove"
                    result = "accepted"
                else:
                    trail = "remove"
                    result = "refused"
            if self.trail3 == 1:
                if self._add == 1:
                    trail = "add"
                    result = "accepted"
                else:
                    trail = "add"
                    result = "refused"

            ast = {1: "*", 0: ""}[self.force_consistent]
            args = (name, self.nsteps, T[0], T[1], T[2], T[3], T[4], T[5],
                    e, ast, fmax, self.temperature, trail, result,
                    self.ad.symbols, self.adsnum, self.u, self.E_applied)
            msg = "%s:  %3d %04d/%02d/%02d %02d:%02d:%02d %15.6f%1s %12.4f %18dK" \
                  " %12s %12s %15s %12d %10.6f %10.3f\n" % args
            self.logfile.write(msg)
            self.logfile.flush()

class GCMC(GCMonteCarlo):
    defaults = {**Optimizer.defaults, 'dr': 0.25, "t": 298 * kB}

    def __init__(self, atoms, ads, calc, u_ref=None, label=None,
                 logfile='gcmc.log', dr=None, displace_percentage=0.25,
                 temperature_K=None, optimizer=FIRE,
                 opt_fmax=None, opt_steps=None, rmin=1.2, rmax=3.0,
                 trajectory=None, seed=None, save_opt=False, save_refused=False, save_all=False,
                 displace=True, add=True, remove=True, steps=0, magmom=None, hight=None, potential_config=None,
                 only_surface=True, bulk='Cu2O'):
        """Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        ads: Atoms object
            The Atoms object need to be add on the atoms.

        calc: Calculator object
            In order to calculate forces and energies,
            you need to attach a calculator object to your atoms object

        u_ref : float
            The reference chemical potential of adsorbates.

        logfile: string
            Text file used to write summary information.

        dr: float
            Atomic displace step, default to 0.25.

        optimizer: optimize algorithm
            Default to use the FIRE to optimize the structure when accept add atom to the system.

        temperature_K: temperature
            Defaults to None. If users don't set value, the optimizer will use 298K

        trajectory: string
            Pickle file used to store trajectory of atomic movement.
            Attention: Due to the algorithm calls the SinglePointCalculator to storage the energy
            and the force of each step, the parameter don't need to be written
            (see gcmc.traj & all_trajectory.traj).
            One can track each GCMC step by looking at the gcmc.traj and when the algorithm finished,
            all movement include optimize will be written.

        seed: digital
            For a parallel calculation, it is important to use the same seed on all processors!

        save_opt: boolean
            Default to False. If set to True, the structure optimize trajectory will be restored.

        """
        restart = None
        master = None
        Optimizer.__init__(self, atoms, restart, logfile, master)

        global L, l, nano
        np.random.seed(seed=seed)

        self.traj = []
        self.sa = []
        self.sr = []
        self.all_traj = []
        self.tags_list = []
        self.save_all_step = []
        self.save_all = save_all
        self.atoms = atoms
        self.ads = self.transfer_ads(ads)
        self.calc = calc
        self.u_ref = u_ref
        self.optimizer = optimizer
        self.save_opt = save_opt
        self.dp = displace_percentage
        self.save_refused = save_refused
        self.rmin = rmin
        self.rmax = rmax
        self.displace = displace
        self.add = add
        self.remove = remove
        self.ad, self.u, _ = self.get_ad_and_uref()
        self.magmom = magmom
        self.steps = steps
        self.hight = hight
        self.potential_config = potential_config
        self.E_applied = None
        self.only_surface = only_surface
        self.bulk = bulk
        self.surf_mask = []

        if label is None:
            self.label = "default"
        elif label == "vasp":
            self.label = "vasp"
            self.vasp = copy.deepcopy(calc)
            self.vasp.set(nsw=steps)
        else:
            self.label = "default"

        if trajectory is None:
            self.Trajectory = "gcmc.traj"
        else:
            self.Trajectory = trajectory

        if dr is None:
            self.dr = self.defaults['dr']
        else:
            self.dr = dr

        if temperature_K is None:
            self.t = self.defaults['t']
            self.beta = 1 / self.t
            self.temperature = 298
        else:
            self.t = temperature_K * kB
            self.beta = 1 / self.t
            self.temperature = temperature_K
        if opt_fmax is None:
            self.opt_fmax = 0.1
        else:
            self.opt_fmax = opt_fmax
        if opt_steps is None:
            self.opt_steps = 100
        else:
            self.opt_steps = opt_steps
        if self.remove and not self.add:
            raise ValueError("In GCMC, remove particle must be associated with add particle. ")

        # set initial tag. tag the MC atoms from 111
        self.tags = 111
        for atom in self.atoms:
            if atom.tag:
                atom.tag = self.tags
                self.tags += 1

        if self.potential_config:
            self.initial_constant_potential()

    def step(self):
        self.E_applied = self.get_E_applied(self.atoms)
        if self.only_surface:
            self.surf_mask = get_surface_atom_GCN(self.atoms)
        if self.displace and not self.add and not self.remove:
            self.atoms = self.trail_displace()
        elif self.displace and self.add and not self.remove:
            if rand() < 0.5:
                self.atoms = self.trail_displace()
            else:
                self.atoms = self.trail_add()
        elif self.displace and self.add and self.remove:
            roll = rand()
            if roll < 1/3:
                self.atoms = self.trail_displace()
            elif roll > 2/3:
                self.atoms = self.trail_add()
            else:
                self.atoms = self.trail_remove()
        elif self.add and not self.displace and not self.remove:
            self.atoms = self.trail_add()
        elif self.add and self.remove and not self.displace:
            if rand() < 0.5:
                self.atoms = self.trail_add()
            else:
                self.atoms = self.trail_remove()

        self.save_all_traj()
        self.dump((self.atoms))

    def initial_constant_potential(self):
        """
        initial constant potential MC.
        """
        self.UvsRHE = self.potential_config.get('URHE', 0)
        pH = self.potential_config.get('pH', 0)
        self.UvsSHE = self.UvsRHE - kB * self.temperature * pH * math.log(10) # UvsSHE= UvsRHE –kBT ln(10)pH/e
        Upzc_config = self.potential_config.get('Upzc_net_config')
        ckpoint = self.potential_config.get('Upzc_net')
        self.pzc_predictor = Upzc_predictor(Upzc_config, ckpoint)

        self.E_applied = self.get_E_applied(self.atoms)
        self.atoms = self.reset_E_applied(self.atoms)

    def reset_E_applied(self, atoms):
        for atom in atoms:
            atom.magmom = [0, 0, self.E_applied]
        return atoms

    def get_E_applied(self, atoms_):
        atoms = atoms_.copy()
        Upzc = self.pzc_predictor.get_Upzc(atoms)
        USHE = self.potential_config.get('USHE', 4.4)
        vp = self.potential_config.get('vaccum_permittivity', 8.85E-12)
        dc = self.potential_config.get('dielectric_constant', 2)
        CH = self.potential_config.get('capacitance', 40)
        E_applied = CH * (self.UvsSHE - (Upzc - USHE)) / dc / vp * 1e-12    # eV/A
        # E = CH * (UvsSHE -Upzc vs SHE) / dielectric_constant/ vaccum_permittivity
        return E_applied

    def get_ad_and_uref(self, tag=None):
        add_index = 0
        if len(self.ads) == 1:
            self.ad = self.ads[0]
            self.u = self.u_ref[0]
            add_index = 0
        else:
            if tag is None:
                self.ad = np.random.choice(self.ads)
                for i, a in enumerate(self.ads):
                    if a == self.ad:
                        self.u = self.u_ref[i]
                        add_index = i
            else:
                self.ad = Atoms()
                for atom in self.atoms:
                    if atom.tag == tag:
                        self.ad.append(atom)
                for i, ad in enumerate(self.ads):
                    if self.ad == ad:
                        self.u = self.u_ref[i]
                        add_index = i

        return self.ad, self.u, add_index

    def get_revised_volume(self):
        """
        To accelerate the simulation,a bias into the simulation, which performs a relaxation on the
        structure after each MC step. This bias can be balanced by revising the volume to effective
        volume.
        In our case, only top side of the slab can perform MC step. And MC step only happens within
        2A above the top atom.
        ****** only for monoclinic system
        :return: effective volume, A^3
        """
        zmin, zmax = self.get_region_boundry(self.atoms)
        region_volume = self.atoms.cell.volume * (zmax - zmin) / self.atoms.cell[2][2]
        atoms_inside = [atom for atom in self.atoms if atom.tag !=0]
        radii = [data.covalent_radii[data.atomic_numbers[atom.symbol]] for atom in atoms_inside]
        occupied = np.sum([4./3.*np.pi*r**3 for r in radii])
        volume_eff = region_volume - occupied
        return volume_eff

    def get_de_broglie_wavelength_cubic(self):
        """
        de Broglie wavelength is calculated as h/(2*pi*m*kB*T)^0.5
        :return:cubic of de Broglie wavelength , Angstrom^3  (A^3)
        return as list consist with self.ads
        """
        wavelength_cubic_list = []

        for ad in self.ads:
            m = ad.get_masses().sum()
            wavelength = _hplanck*J*s / math.sqrt(2*pi*m*kB*298.15)
            wavelength_cubic_list.append(math.pow(wavelength, 3))
        return wavelength_cubic_list

    def get_region_boundry(self, atoms):
        """
        :param atoms:
        MC effective region changes with MC step.
        the effective region is above the slab and below the height_limit
        only for monoclinic system
        :return: zmin, zmax
        """

        z_position = []
        for index, atom in enumerate(atoms):
            if not self.only_surface:
                z_position.append(atom.position[2])
            elif self.only_surface:
                if self.surf_mask[index]:
                    z_position.append(atom.position[2])
        zmin = np.min(z_position, axis=0)
        zmax = np.max(z_position, axis=0)
        if self.hight:
            zmin, zmax = max(self.hight[0], zmin + 2), min(zmax + 2, self.hight[1])
        return zmin, zmax

    def trail_add(self):
        self.trail1 = 0
        self.trail2 = 0
        self.trail3 = 1
        self._add = 0
        Eo = self.atoms.get_potential_energy()
        self.ad, self.u, add_index = self.get_ad_and_uref()
        new = self.add_particle(ads=self.ad)
        new = self.reset_E_applied(new)
        add = self.add_judgement(new)
        if add:
            En = self.get_energy(new)
            V = self.get_revised_volume()
            wavelength_cubic = self.get_de_broglie_wavelength_cubic()[add_index]
            # coef = V / wavelength_cubic / (len(new))
            coef = 1
            if En - Eo - self.u < 0:
                new = self.structure_opt(new)
                self.atoms = new
                self._add = 1
                self.adsnum = self.get_adsnum(self.atoms)
                self.save_accepted_atoms(new)
            elif rand() < min(1, coef * np.exp(-(En - Eo - self.u) * self.beta)):
                new = self.structure_opt(new)
                self.atoms = new
                self._add = 1
                self.adsnum = self.get_adsnum(self.atoms)
                self.save_accepted_atoms(new)
            else:
                self._add = 0
                self.save_refused_atoms(new)
        return self.atoms

    def add_particle(self, ads, **kwargs):
        atoms = self.atoms.copy()
        ads = ads.copy()
        ads.set_tags(self.tags)

        zmin, zmax = self.get_region_boundry(atoms)

        x_scale = np.random.uniform(0, 1)
        y_scale = np.random.uniform(0, 1)
        z = np.random.uniform(zmin, zmax)
        x, y = x_scale * self.atoms.cell[0][:2] + y_scale * self.atoms.cell[1][:2]

        ads.translate([x, y, z] - ads.positions[0])
        for ad in ads:
            atoms.append(ad)
        self.tags += 1

        return atoms

    def add_judgement(self, atoms):
        # only one atom is added
        d = atoms.get_distances(len(atoms) - 1, indices=[i for i in range(len(atoms)-1)], mic=True)
        add = min(d) >= self.rmin
        return add

    def trail_remove(self):
        self.trail1 = 0
        self.trail2 = 1
        self.trail3 = 0
        self._rem = 0
        Eo = self.atoms.get_potential_energy()
        tags = self.get_tag(self.atoms)
        if self.only_surface:
            surf_mask = self.surf_mask
            surface_atom_tags = self.atoms[surf_mask].get_tags()
            tags_ = [tag for tag in tags if tag in surface_atom_tags]
            tags = tags_

        if tags:
            tag = np.random.choice(tags)
            # randomly remove taged atoms maybe
            # we can further point out surface atoms to remove
        else:
            tag = None
        self.ad, self.u, ad_index = self.get_ad_and_uref(tag)
        new = self.remove_particle(tag=tag)
        new = self.reset_E_applied(new)
        En = self.get_energy(new)
        compare = self.compare_atoms(self.atoms, new)
        if not compare:
            V = self.get_revised_volume()
            wavelength_cubic = self.get_de_broglie_wavelength_cubic()[ad_index]
            # coef = wavelength_cubic * (len(self.atoms)) / V
            coef = 1
            if En - Eo + self.u < 0:
                new = self.structure_opt(new)
                self.atoms = new
                self._rem = 1
                self.adsnum = self.get_adsnum(self.atoms)
                self.save_accepted_atoms(new)
            elif rand() < min(1, coef * np.exp(-(En - Eo + self.u) * self.beta)):
                new = self.structure_opt(new)
                self.atoms = new
                self._rem = 1
                self.adsnum = self.get_adsnum(self.atoms)
                self.save_accepted_atoms(new)
            else:
                self._rem = 0
                self.save_refused_atoms(new)
        return self.atoms

    def remove_particle(self, tag=None):
        atoms = self.atoms.copy()
        atoms.calc = self.calc
        if tag:
            atoms.pop(np.where(atoms.get_tags() == tag)[0].item())
            return atoms
        else:
            return atoms

    def trail_displace(self):
        self.trail1 = 1
        self.trail2 = 0
        self.trail3 = 0
        self._dis = 0
        Eo = self.atoms.get_potential_energy()
        new = self.move_atom()
        if self.magmom:
            new = self.reset_E_applied(new)
        En = self.get_energy(new)
        if En < Eo:
            new = self.structure_opt(new)
            self.atoms = new
            self._dis = 1
            self.save_accepted_atoms(new)
        elif rand() < np.exp(-(En - Eo) * self.beta):
            new = self.structure_opt(new)
            self.atoms = new
            self._dis = 1
            self.save_accepted_atoms(new)
        else:
            self._dis = 0
            self.save_refused_atoms(new)

        return self.atoms

    def displace_judgement(self, atoms):
        # in case extrame close
        min_d = []
        for i in range(len(atoms)):
            d = atoms.get_distances(i, indices=[ii for ii in range(len(atoms)) if ii != i], mic=True)
            min_d.append(min(d))
        displace = min(min_d) >= self.rmin
        return displace

    def move_atom(self, L=None, cm=True):
        atoms = self.atoms.copy()
        image = self.atoms.copy()
        if L is None:
            L = []
        for atom in atoms:
            L.append(atom.index)
        l = np.random.choice(L, size=ceil((len(L) * self.dp)), replace=False)
        for index in l:
            atoms[index].position = atoms[index].position + \
                                    self.dr * np.random.uniform(-1., 1., (1, 3))
        rn = atoms.get_positions()
        image.set_positions(rn)
        return image
