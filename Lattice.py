from functools import reduce
import itertools as it
import numpy as np
from BlochVector import BlochVector


class Lattice:
    def __init__(self, unit_cell, N, *args, **kwargs):
        self.unit_cell = unit_cell
        self.N = np.array(N)
        self.Np = reduce(lambda a, b: a*b, self.N, 1)
        self.M = self.unit_cell.num_particles
        self.dim_space = len(self.N)

    def Np(self):
        return self.Np

    def unit_cells(self):
        ranges = [range(n) for n in self.N]
        return (np.array(a) for a in it.product(*ranges))

    def cell_displacement(self, p1, p2):
        dp0 = p1 - p2
        dp1 = np.empty_like(dp0)
        for di, Ni, i in zip(dp0, self.N, it.count()):
            if abs(di) > 1 and abs(di) == Ni - 1:
                dp1[i] = -np.sign(di)
            else:
                dp1[i] = di
        return dp1

    def are_connected(self, kappa1, p1, kappa2, p2):
        """Returns true if there is a connection between particle number k1
        in unit cell p1 and particle number k2 in unit cell p2
        :param kappa1: Index of particle 1 in unit cell
        :param p1: Position index (nx, ny) of unit cell 1
        :param kappa2: Index of particle 2 in unit cell
        :param p2: Position index (nx, ny) of unit cell 2
        """
        disp = self.cell_displacement(p2, p1)
        is_neg_neighbor = reduce(
            lambda a, b: a and b, map(lambda x: x == 0 or x == -1, disp), True)
        is_pos_neighbor = reduce(
            lambda a, b: a and b, map(lambda x: x == 0 or x == 1, disp), True)
        if is_pos_neighbor:
            return self.unit_cell.connected(kappa1, kappa2, disp)
        elif is_neg_neighbor:
            return self.are_connected(kappa2, p2, kappa1, p1)
        else:
            return False

    def displacement_mod_a(self, kappa1, p1, kappa2, p2):
        kdisp = self.unit_cell.displacement_mod_a(i=kappa1, j=kappa2)
        pdisp = self.cell_displacement(p1=p1, p2=p2)
        return kdisp + pdisp

    def displacement(self, kappa1, p1, kappa2, p2):
        return np.dot(
            self.unit_cell.a_matrix,
            self.displacement_mod_a(kappa1=kappa1, p1=p1, kappa2=kappa2, p2=p2)
        )

    # TODO: fix so that opposite edge is included
    # def adjacent_cells(self, p):
    #     for cell_disp in it.product([-1, 0, 1], repeat=self.dim_space):
    #         yield p + np.array(cell_disp)

    def q_vectors(self):
        for m in self.unit_cells():
            yield BlochVector(m, self.N)

    def _particles(self):
        return it.product(range(self.M), range(self.dim_space))

