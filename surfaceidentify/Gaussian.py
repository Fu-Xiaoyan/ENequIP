import ase
import itertools
import operator
import ase.io
from ase.visualize import view

from ase import neighborlist
import math

import numpy as np
def Fc_cos(Rij, Rc):
    '''
    cutoff function
    :return: 
    '''
    return 0.5 * (1 + math.cos(Rij/Rc*math.pi))


def G1(atoms, ad_index, cutoff=6):
    elements = ['Cu', 'O']
    G1_eta = 1
    G1_Rs = 0.5
    i, j, d = neighborlist.neighbor_list('ijd', atoms, cutoff)
    neighbor_index = j[i == ad_index]
    neighbor_dis = d[i == ad_index]
    g1 = {}
    for e in elements:
        g1[e] = 0
    for jj, index in enumerate(neighbor_index):
        e = atoms[index].symbol
        g1[e] += math.exp(-G1_eta*(neighbor_dis[jj]-G1_Rs) ** 2) * Fc_cos(neighbor_dis[jj], cutoff)

    return g1

def G2(atoms, ad_index, cutoff=6):
    """
    G4 in Behler 2015
    :param atoms:
    :param ad_index:
    :param cutoff:
    :return:
    """
    elements = ['Cu', 'O']
    G2_eps = 0.5
    G2_eta = 0.15
    G2_lambda = 1
    comb_permu = list(itertools.combinations(elements, 2))
    self_permu = [(e, e) for e in elements]
    elements_permu = comb_permu + self_permu
    g2 = {}
    for permu in elements_permu:
        g2[permu] = 0

    i, j, d, D = neighborlist.neighbor_list('ijdD', atoms, cutoff)

    neighbor_index = j[i == ad_index]

    di = d[i == ad_index]
    Di = D[i == ad_index]

    angular_permu = list(itertools.combinations(neighbor_index, 2))

    for j_index, k_index in angular_permu:
        permu = None
        for comb in elements_permu:
            if operator.eq(comb, (atoms[j_index].symbol, atoms[k_index].symbol)) or \
                    operator.eq(comb, (atoms[k_index].symbol, atoms[j_index].symbol)):
                permu = comb
                break
        else:
            print('Angular Gaussian Failed')
            raise ValueError()
        Dij = Di[list(neighbor_index).index(j_index)]
        Dik = Di[list(neighbor_index).index(k_index)]
        dij = di[list(neighbor_index).index(j_index)]
        dik = di[list(neighbor_index).index(k_index)]
        # djk = atoms.get_distance(j_index, [k_index], mic=True)
        cos_theta = np.dot(Dij, Dik) / dij / dik
        if cos_theta < -1:  # incase of nan
            g2[permu] += 0
        else:
            g2[permu] += (1 + G2_lambda * cos_theta) ** G2_eps \
                         * math.exp(-G2_eta * (dij**2+dik**2)) \
                         * Fc_cos(dij, cutoff) * Fc_cos(dik, cutoff)
    for k, g in g2.items():
        g2[k] = g * 2 ** (1 - G2_eps)
    return g2
