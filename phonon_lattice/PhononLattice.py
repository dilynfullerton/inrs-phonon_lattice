import itertools as it
import numpy as np
import qutip
from numpy import array, exp, sqrt, pi, dot
from numpy.linalg import norm, eigh, inv
from scipy.linalg import sqrtm
from .BlochVector import BlochVector


class PhononLattice:
    def __init__(self, unit_cell, N, c_matrix, nfock=2):
        self.unit_cell = unit_cell
        self.N = array(N)
        self.Np = np.product(self.N)
        self.dim = len(self.N)
        self.M = self.unit_cell.num_particles
        self._nfock = nfock
        self.c_matrix = c_matrix
        self._evals_dict = dict()
        self._evect_dict = dict()
        self._indices = list(
            it.product(range(self.M), range(self.dim))
        )

    def num_connections(self, k, p):
        return self.unit_cell.num_connections(k)

    def are_connected(self, k1, p1, k2, p2):
        """Returns true if there is a connection between particle number k1
        in unit cell p1 and particle number k2 in unit cell p2
        :param k1: Index of particle 1 in unit cell
        :param p1: Position index (nx, ny) of unit cell 1
        :param k2: Index of particle 2 in unit cell
        :param p2: Position index (nx, ny) of unit cell 2
        """
        p1 = array(p1)
        p2 = array(p2)
        disp = p1 - p2
        if norm(disp, ord=1) == 0:  # internal
            return self.unit_cell.connected_int(k1, k2)
        for i in range(len(self.N)):
            if p1[i] - p2[i] == 1 or p2[i] - p1[i] == self.N[i] - 1:
                return self.unit_cell.connected_ext(k2, k1, axis=i)
            elif p2[i] - p1[i] == 1 or p1[i] - p2[i] == self.N[i] - 1:
                return self.unit_cell.connected_ext(k1, k2, axis=i)
        else:
            return False

    def position(self, k, p):
        return self.unit_cell.position(k, p)

    def displacement(self, k1, p1, k2, p2):
        return self.position(k2, p2) - self.position(k1, p1)

    def unit_cells(self):
        ranges = [range(n) for n in self.N]
        return (array(a) for a in it.product(*ranges))

    def d_matrix(self, k1, x1, k2, x2):
        def _d(q):
            d = 0
            for p in self.unit_cells():
                dp = self.c_matrix(self, k1, x1, np.zeros_like(p), k2, x2, p)
                dp *= exp(1j * 2*pi * q.dot(p))
                d += dp
            return d * 1/sqrt(self.unit_cell.mass(k1) * self.unit_cell.mass(k2))
        return _d

    def e(self, k, x, v):
        i = self._indices.index((k, x))

        def _e(q):
            if q not in self._evals_dict:
                self._set_d_eigenvectors(q)
            return self._evect_dict[q][v][i]
        return _e

    def omega2(self, q, v):
        if q not in self._evals_dict:
            self._set_d_eigenvectors(q)
        return self._evals_dict[q][v]

    def q_vectors(self):
        for m in self.unit_cells():
            vect = [mi/ni for mi, ni in zip(m, self.N)]
            yield BlochVector(vect)

    def _A(self):
        for q in self.q_vectors():
            if q == -q:
                yield q

    def _B(self):
        b = []
        for q in self.q_vectors():
            if q != -q and -q not in b:
                b.append(q)
                yield q

    def operator_q_vectors(self):
        return it.chain(self._A(), self._B())

    def annihilation_operator(self, q, v):
        if q in self._A():
            x, px = self._x(q, v)
            return 1/sqrt(2) * (x + 1j*px)
        elif q in self._B():
            x, px = self._x(q, v)
            y, py = self._y(q, v)
            return 1/2 * (x + 1j*px) + 1j/2 * (y + 1j*py)

    def _ops(self, k, x, p, op):
        dim = self.dim * self.M * self.Np
        p = array(p)
        xop = [qutip.qeye(self._nfock)] * dim
        idx = x + self.dim * k + self.dim * self.M * p[0]
        for i in range(self.dim-1):
            idx += self.dim * self.M * self.N[i] * p[i+1]
        xop.insert(idx, op)
        xop.pop(idx+1)
        return qutip.tensor(xop)

    def _x_ops(self, k, x, p):
        return self._ops(k, x, p, op=qutip.position(self._nfock))

    def _p_ops(self, k, x, p):
        return self._ops(k, x, p, op=qutip.momentum(self._nfock))

    def _z(self, q, v):
        real_z = 0+0j
        imag_z = 0+0j
        real_pz = 0+0j
        imag_pz = 0+0j
        for k, x, p in it.product(range(self.M), range(self.dim), self.unit_cells()):
            zi = exp(-1j * 2*np.pi * (q.dot(p)))
            zi *= sqrt(self.unit_cell.mass(k)/self.unit_cell.mass(0))
            zi *= np.conj(self.e(k, x, v)(q)) / sqrt(self.Np)
            xop = self._x_ops(k, x, p)
            pop = self._p_ops(k, x, p)
            real_z += zi.real * xop
            imag_z += zi.imag * xop
            real_pz += zi.real * pop
            imag_pz += zi.imag * pop
        return real_z, imag_z, real_pz, imag_pz

    def _l(self, q, v):
        m0 = self.unit_cell.mass(0)
        omega2 = self.omega2(q, v)+0j
        return 1 / sqrt(2 * m0 * sqrt(omega2))

    def _x(self, q, v):
        real_z, imag_z, real_pz, imag_pz = self._z(q, v)
        l = self._l(q, v)
        if q in self._A():
            return real_z/2/l, real_pz/2/l
        elif q in self._B():
            return real_z/l, real_pz/l

    def _y(self, q, v):
        real_z, imag_z, real_pz, imag_pz = self._z(q, v)
        l = self._l(q, v)
        if q in self._B():
            return imag_z / l, imag_pz / l

    def _get_matrix_rep_d(self, q):
        dim = self.M * self.dim
        d_mat = np.empty(shape=(dim, dim), dtype=np.complex)
        for k1x1, i in zip(it.product(range(self.M), range(self.dim)),
                           it.count()):
            k1, x1 = k1x1
            for k2x2, j in zip(
                    it.product(range(k1, self.M), range(x1, self.dim)),
                    it.count()
            ):
                k2, x2 = k2x2
                d_mat[i, j] = self.d_matrix(k1, x1, k2, x2)(q)
                if i != j:
                    d_mat[j, i] = np.conj(d_mat[i, j])
        return d_mat

    def _orthonormal_eigenvectors(self, dmat):
        evals, evects = eigh(a=dmat)
        on_evects = dot(evects, inv(sqrtm(dot(evects.conj().T, evects))))
        return evals, [on_evects[:, i] for i in range(len(evals))]

    def _set_d_eigenvectors(self, q):
        evals, evects = self._orthonormal_eigenvectors(
            dmat=self._get_matrix_rep_d(q))
        self._evals_dict[q] = evals
        self._evect_dict[q] = evects


class PhononLattice1D(PhononLattice):
    def __init__(self, unit_cell, N_x, c_matrix, n_fock=2):
        super(PhononLattice1D, self).__init__(
            unit_cell=unit_cell,
            N=[N_x],
            c_matrix=c_matrix,
            nfock=n_fock
        )
        self.Nx = N_x


class PhononLattice2D(PhononLattice):
    def __init__(self, unit_cell, N_x, N_y, c_matrix, n_fock=2):
        super(PhononLattice2D, self).__init__(
            unit_cell=unit_cell,
            N=[N_x, N_y],
            c_matrix=c_matrix,
            nfock=n_fock
        )
        self.Nx = N_x
        self.Ny = N_y


class PhononLattice3D(PhononLattice):
    def __init__(self, unit_cell, N_x, N_y, N_z, c_matrix, n_fock=2):
        super(PhononLattice3D, self).__init__(
            unit_cell=unit_cell,
            N=[N_x, N_y, N_z],
            c_matrix=c_matrix,
            nfock=n_fock
        )
        self.Nx = N_x
        self.Ny = N_y
        self.Nz = N_z