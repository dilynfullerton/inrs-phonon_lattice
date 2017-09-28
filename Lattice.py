from functools import reduce
import itertools as it
import numpy as np
from BlochVector import BlochVector


class Lattice:
    def __init__(self, unit_cell, N, *args, **kwargs):
        self.unit_cell = unit_cell
        self.N = np.array(N, dtype=np.int)
        self.Np = reduce(lambda a, b: a*b, self.N, 1)
        self.M = self.unit_cell.num_particles
        self.dim_space = len(self.N)
        self.p0 = np.zeros_like(self.N)

    def Np(self):
        """The number of unit cells in the supercell
        """
        return self.Np

    def unit_cells(self):
        """Returns a generator of unit cells p = [n1, n2, n3, ...],
        specifying the periodic cell displacement from p0 = [0, 0, 0, ...].
        """
        ranges = [range(n) for n in self.N]
        return (np.array(a) for a in it.product(*ranges))

    def are_adjacent(self, p1, p2):
        periodic_disp = self.periodic_displacement_mod_a(p1, p2)
        return max(periodic_disp) <= 1 and min(periodic_disp) >= -1

    def adjacent_cells(self, p):
        for p1 in self.unit_cells():
            if self.are_adjacent(p, p1):
                yield p1

    def are_connected(self, kappa1, p1, kappa2, p2):
        """Returns true if there is a connection between particle number k1
        in unit cell p1 and particle number k2 in unit cell p2
        :param kappa1: Index of particle 1 in unit cell
        :param p1: Position index (nx, ny) of unit cell 1
        :param kappa2: Index of particle 2 in unit cell
        :param p2: Position index (nx, ny) of unit cell 2
        """
        disp = self.periodic_displacement_mod_a(p2, p1)
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

    def periodic_displacement_mod_a(self, p1, p2):
        """Returns the unique vector dp, such that
                p2_i + dp_i = p1_i (mod N_i), and
                -1/2 < dp_i/N_i <= 1/2
        """
        disp_abs = p1 - p2
        disp_rel = np.mod(disp_abs, self.N)
        for di, i in zip(disp_rel, it.count()):
            if di > self.N[i] / 2:
                disp_rel[i] = di - self.N[i]
        return disp_rel

    def periodic_particle_displacement_mod_a(self, kappa1, p1, kappa2, p2):
        kdisp = self.unit_cell.particle_displacement_in_cell_mod_a(
            i=kappa1, j=kappa2)
        return self.periodic_displacement_mod_a(p1=p1, p2=p2-kdisp)

    def periodic_displacement_distance(self, kappa1, p1, kappa2, p2):
        return np.dot(
            self.unit_cell.a_matrix,
            self.periodic_particle_displacement_mod_a(
                kappa1=kappa1, p1=p1, kappa2=kappa2, p2=p2)
        )

    def q_vectors(self):
        for m in self.unit_cells():
            yield BlochVector(m, self.N)

    def _particles(self):
        return it.product(range(self.M), range(self.dim_space))

