# from Lattice import Lattice
# from AbsHamiltonian import *
# from PhononLattice import PhononLattice
# from qutip import sigmam, qeye, tensor
#
#
# class ElectronLatticeSimple(Lattice, AbsElectronHamiltonian):
#     def __init__(self, unit_cell, N, m_e, epsilon_F, *args, **kwargs):
#         """Creates a single-particle electron lattice based on the simple
#         expression for epsilon.
#         :param m_e: Electron mass
#         :param epsilon_F: Fermi energy
#         """
#         super(ElectronLatticeSimple, self).__init__(
#             unit_cell=unit_cell, N=N, *args, **kwargs)
#         self.m_e = m_e
#         self.epsilon_F = epsilon_F
#         self._k_vectors = [k for k in self._iter_k()]
#         self._num_k = len(self._k_vectors)
#         self._num_n = len(self._iter_n())
#
#     def epsilon(self, n, k):
#         return abs(k)**2 / 2 / self.m_e - self.epsilon_F
#
#     def c(self, n, k):
#         ops = [qeye(2)] * self._num_n * self._num_k
#         idx = self._k_vectors.index(k) + self._num_k * n
#         ops.insert(idx, sigmam())
#         ops.pop(idx+1)
#         return tensor(ops)
#
#     def _iter_k(self):
#         return self.q_vectors()
#
#     def _iter_n(self):
#         return range(self.dim_space * self.M)
#
#
# class ElectronPhononLatticeSimple(ElectronLatticeSimple, PhononLattice):
#     def __init__(self, unit_cell, N, m_e, epsilon_F, c_matrix, n_fock,
#                  force_adj_only, *args, **kwargs):
#         super(ElectronPhononLatticeSimple, self).__init__()
#
#     def N_p(self):
#         pass
#
#     def g_DW(self, m, n, v1, v2):
#         pass
#
#     def g(self, m, n, v):
#         pass
