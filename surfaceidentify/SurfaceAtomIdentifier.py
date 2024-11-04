from pymatgen.io.ase import AseAtomsAdaptor
from .generalized_CN import GeneralizedCN
from .Gaussian import G1, G2
from pymatgen.analysis.local_env import VoronoiNN
import ase.io
import numpy as np
# from ovito.pipeline import StaticSource, Pipeline
# from ovito.io.ase import ase_to_ovito
# from ovito.modifiers import ConstructSurfaceModifier


def get_coordinate_number_of_local_atoms(atoms, center_atom_index,bulk, ad_metal, cutoff=6, neighbor_number=4):
    voronoi_nn = VoronoiNN(cutoff=4, tol=0.1)  # 0.1 chosen for better detection
    # get neighbor index
    distance = atoms.get_distances(center_atom_index, [i for i in range(len(atoms))], mic=True)
    distance_indexed = [(i, distance[i]) for i in range(len(atoms))]
    sorted_d = sorted(distance_indexed, key=lambda x : x[1])
    neighbor_list = [i[0] for i in sorted_d[1:neighbor_number+1]]
    surface_struct = AseAtomsAdaptor.get_structure(atoms)
    voronoi_cn_list = [voronoi_nn.get_cn(surface_struct, neighbor, use_weights=True) for neighbor in neighbor_list]
    cn_sum =sum(voronoi_cn_list)
    return cn_sum


def get_cooridinate_number_and_gaussian_FP(atoms, atom_index, bulk, ad_metal='Cu', cutoff=6):
    cn = get_coordinate_number_of_local_atoms(atoms, atom_index,bulk, ad_metal, cutoff)
    g1 = G1(atoms, ad_index=atom_index, cutoff=cutoff)
    g2 = G2(atoms, ad_index=atom_index, cutoff=cutoff)
    return g1['Cu'], g1['O'], g2[('Cu', 'Cu')], g2[('Cu', 'O')], g2[('O', 'O')], cn


def get_surface_atom_GCN(atoms, tol1=3, tol2=0.5, cluster=False):
    """
    identify surface atoms using (generalized) coordination number
    for copper atoms, CN < 12 is surface atom
    for oxygen atoms, GCN < 4 is surface atom
    :param atoms: Atoms (ase.atoms) An object of the structure.
    tol1: tolerance for surface Cu identification
    tol2: tolerance for surface O identification
    cluster: if not cluster, only upper surface atoms are labeled
    :return:(list): A list of 0's and 1's whose indices align with the atoms in
            surface_struct. 0's indicate a bulk atom and 1 indicates a surface atom.
    """
    gcn = GeneralizedCN(atoms)
    tags = [False for i in range(len(atoms))]
    center_of_mass = atoms.get_center_of_mass(scaled=True)
    scaled_position = atoms.get_scaled_positions(wrap=True)
    for i in range(len(atoms)):
        # surface atoms should higher than the center of geometric
        if scaled_position[i][2] < center_of_mass[2] and not cluster:
            continue
        if atoms[i].symbol == 'Cu':
            # CN = gcn.get_elementary_CN(i)
            Cu_GCN, O_GCN, GCN = gcn.get_GCN(i)
            CN = gcn.get_elementary_CN(i)
            if GCN + tol1 < 12 and CN < 10:
                tags[i] = True
        elif atoms[i].symbol == 'O':
            Cu_GCN, O_GCN, GCN = gcn.get_GCN(i)
            if GCN + tol2 < 4:
                tags[i] = True
    return tags
