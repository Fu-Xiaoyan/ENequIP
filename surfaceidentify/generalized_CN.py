import numpy as np

# generalized coordination number
# generalized oxygen number

class GeneralizedCN():
    """
    GCN = sum( CN_i / CNmax_i )   i is neighbor atoms
    for O, CNmax = 4

    """
    def __init__(self, atoms):
        self.atoms = atoms
        self.di = {
            'O': 0.63,
            'Cu': 1.12,
        }
        self.delta = 0.45
        self.CuO_double_bond = 1.89
        self.distances = atoms.get_all_distances(mic=True)
        self.get_adjacency_matrix()
        self.get_weight_matrix()


    def if_connected(self, atom1, atom2):
        d_max = self.di[atom1.symbol] + self.di[atom2.symbol] + self.delta
        dij = self.distances[atom1.index][atom2.index]
        connection = 0
        if dij <= d_max:
            connection = 1
        return connection

    def get_adjacency_matrix(self):
        atom_number = len(self.atoms)
        ad_matrix = np.zeros((atom_number, atom_number))
        for i in range(atom_number):
            for j in range(i+1, atom_number):
                ad_matrix[i][j] = self.if_connected(self.atoms[i], self.atoms[j])
        ad_matrix += ad_matrix.T
        self.adjac_matrix = ad_matrix

    def get_CN(self, atom_index):
        return sum(self.adjac_matrix[atom_index])

    def get_neighbor_index(self, atom_index):
        neighbor_list = [i for i in range(len(self.atoms)) if self.adjac_matrix[atom_index][i] == 1]
        return neighbor_list

    def get_weight_matrix(self):
        atom_number = len(self.atoms)
        weight_matrix = np.zeros((atom_number, atom_number))
        w = {
            'O': 3,
            'Cu': 1,
        }
        for i in range(atom_number):
            for j in range(i+1, atom_number):
                w_bond = 2 if self.distances[i][j] < self.CuO_double_bond else 1
                weight_matrix[i][j] = w[self.atoms[i].symbol] * w[self.atoms[j].symbol] * w_bond
        weight_matrix += weight_matrix.T
        self.weight_matrix = weight_matrix

    def get_elementary_CN(self, atom_index):
        """
        Cu - Cu bond count 1
        Cu - O bond count 3
        O - O bond count 9
        CN_max = 12
        this will lead to CuO bulk Cu_GCN = 12   O_GCN = 12
        *****   but in Cu2O bulk Cu_GCN =  6  ***********
        so we set Cu - O  < 1.9 A as Cu = O and count 6
        """
        elmentary_CN = np.dot(self.adjac_matrix[atom_index], self.weight_matrix[atom_index])
        return elmentary_CN

    def get_GCN(self, atom_index):
        """
        :param atom_index:
        :return: Cu_GCN, O_GCN
        """
        neighbor_list = self.get_neighbor_index(atom_index)
        neighbor_eCN = [self.get_elementary_CN(i) for i in neighbor_list]
        Cu_mask = [1 if self.atoms[i].symbol == 'Cu' else 0 for i in neighbor_list]
        O_mask = [0 if self.atoms[i].symbol == 'Cu' else 1 for i in neighbor_list]
        Cu_GCN = np.dot(neighbor_eCN, Cu_mask) / 12
        O_GCN = np.dot(neighbor_eCN, O_mask) / 4  # for Cu-O bond, count 3-fold than Cu-Cu bond
        GCN = O_GCN + Cu_GCN
        return Cu_GCN, O_GCN, GCN

    def get_all_GCN1(self):
        GCN_all = []
        for index, atom in enumerate(self.atoms):
            _, _, GCNi = self.get_GCN(index)
            GCN_all.append(GCNi)
        self.gcn = GCN_all

    def get_CNA_adjacency_matrix(self):
        """
        larger cutoff radius for CNA.
        r_cut = (1+2^1/2)/2*sum_6(rij)/6
        """
        atom_number = len(self.atoms)
        self.CNA_adjacency_matrix = np.zeros((atom_number, atom_number))
        for i in range(atom_number):
            distancesi = sorted(self.distances[i])
            r_cut_i = (1 + 2 ** 0.5) / 2 * sum(distancesi[:6]) / 6
            for j in range(atom_number):
                if self.distances[i][j] < r_cut_i:
                    self.CNA_adjacency_matrix[i][j] = 1
                else:
                    self.CNA_adjacency_matrix[i][j] = 0

    def get_edge_number(self, atom_index):
        """
        Number of edges between neighbor nodes.
        From CNA
        :return:
        """
        if not hasattr(self, 'CNA_adjacency_matrix'):
            self.get_CNA_adjacency_matrix()
        adjacency_matrix = self.CNA_adjacency_matrix
        neighbor_list = [i for i in range(len(self.atoms)) if self.adjac_matrix[atom_index][i] == 1]
        edge_number = 0
        for neighbor_index in neighbor_list:
            edge_number += sum(adjacency_matrix[neighbor_index][neighbor_list])
        edge_number = edge_number / 2

        edge_number_normalized = edge_number / 12
        return edge_number_normalized


def test_GCN():
    import ase.io
    from ase.visualize import view
    import os
    # dir = '../../Enequip-bk-20230601/surfaceidentify/test_surface_identify'
    # traj_list = os.listdir(dir)

    # typical
    import pandas as pd
    cluster_center_df = pd.read_excel('../../Cu-based-eNO3RR-计算数据.xlsx', sheet_name='gaussian FP')
    typical_atoms_df = cluster_center_df[12:]
    typical_atoms = typical_atoms_df[['label', 'traj', 'atom_index']].to_numpy()
    traj_list = typical_atoms[:, 1]
    dir = os.getcwd()

    traj = []
    for traj_name in traj_list:
        if 'xlsx' in traj_name:
            continue
        if '.traj' in traj_name:
            atoms = ase.io.read(os.path.join(dir, traj_name), index='-1')
            atoms.set_initial_magnetic_moments(magmoms=None)
        else:
            atoms = ase.io.read(os.path.join(dir,traj_name))
        gcn = GeneralizedCN(atoms)
        for i in range(len(atoms)):
            Cu_GCN, O_GCN, GCN = gcn.get_GCN(i)
            CN = gcn.get_elementary_CN(i)
            Edge_number = gcn.get_edge_number(i)
            atoms[i].charge = GCN
            atoms[i].magmom = Edge_number

        traj.append(atoms)
    view(traj)


if __name__ == '__main__':
    test_GCN()

"""   
Jmol’s autoBond Algorithm

dij <= ri + rj + delta
rj is the elemental radius of the atom at site j
δ is a tolerance parameter fixed at 0.45 Å
https://sourceforge.net/p/jmol/code/HEAD/tree/trunk/Jmol/src/org/jmol/util/Elements.java#l798

(default): Values taken from OpenBabel.
        di = {
            'O': 0.680,
            'Cu': 1.52,
        }

   * http://sourceforge.net/p/openbabel/code/485/tree/openbabel/trunk/data/element.txt (dated 10/20/2004)
   * 
   * These values are a mix of common ion (Ba2+, Na+) distances and covalent distances.
   * They are the default for autobonding in Jmol.
   
   
   
Column 2: Blue Obelisk Data Repository (2/22/2014)

        di = {
            'O': 0.63,
            'Cu': 1.12,
        }
   * https://github.com/wadejong/bodr/blob/c7917225cad829507bdd4c8c2fe7ebd3d795c021/bodr/elements/elements.xml 
   * which is from: 
   * 
   * Pyykkö, P. and Atsumi, M. (2009), 
   * Molecular Single-Bond Covalent Radii for Elements 1–118. 
   * Chem. Eur. J., 15: 186–197. doi: 10.1002/chem.200800987
   * 
   * (See also http://en.wikipedia.org/wiki/Covalent_radius)
   *  
   * These are strictly covalent numbers.
"""