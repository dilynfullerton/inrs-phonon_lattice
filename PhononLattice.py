import itertools as it
import operator
import numpy as np
import qutip
from numpy import exp, sqrt, pi, dot
from scipy.linalg import sqrtm, eigh, inv, norm
from AbsHamiltonian import AbsPhononHamiltonian
from Lattice import Lattice


class PhononLattice(Lattice, AbsPhononHamiltonian):
    def __init__(self, unit_cell, N, c_matrix, nfock=2, force_adj_only=True,
                 *args, **kwargs):
        super(PhononLattice, self).__init__(
            unit_cell=unit_cell, N=N, *args, **kwargs)
        self.num_modes = self.dim_space * self.M
        self._nfock = nfock
        self.c_matrix = c_matrix
        self._evals_dict = dict()
        self._evect_dict = dict()
        self._indices = list(
            it.product(range(self.M), range(self.dim_space))
        )
        # self._force_adj = force_adj_only
        self._force_adj = force_adj_only
        self.A = set(self._gen_A())
        self.B = set(self._gen_B())

    def omega(self, q, v):
        return sqrt(self.omega2(q, v))

    def a(self, q, v):
        if q in self.A:
            x, px = self._x(q, v)
            return 1/sqrt(2) * (x + 1j*px)
        elif q in self.B:
            x, px = self._x(q, v)
            y, py = self._y(q, v)
            return 1/2 * (x + 1j*px) + 1j/2 * (y + 1j*py)

    def _iter_q(self):
        return self.operator_q_vectors()

    def _iter_v(self):
        return range(self.num_modes)

    def b_matrix(self):
        return 2*pi * inv(self.unit_cell.a_matrix.conj().T)

    def b_vectors(self):
        b = self.b_matrix()
        return [b[:, i] for i in range(self.dim_space)]

    def d_matrix(self, kappa1, alpha1, kappa2, alpha2):
        def _d(q):
            d = 0
            for p in self.unit_cells():
                if self._force_adj and not self.are_adjacent(p, self.p0):
                    continue
                dp = self.c_matrix(lattice=self,
                                   kappa1=kappa1, alpha1=alpha1, p1=self.p0,
                                   kappa2=kappa2, alpha2=alpha2, p2=p)
                dp *= exp(1j * 2*pi * q.dot(p))
                d += dp
            d *= 1/sqrt(
                self.unit_cell.mass(kappa1) * self.unit_cell.mass(kappa2))
            return d
        return _d

    def e(self, kappa, alpha, v):
        i = self._indices.index((kappa, alpha))

        def _e(q):
            if q not in self._evals_dict:
                self._set_d_eigenvectors(q)
            return self._evect_dict[q][v][i]
        return _e

    def omega2(self, q, v):
        if q not in self._evals_dict:
            self._set_d_eigenvectors(q)
        return self._evals_dict[q][v]

    def _gen_A(self):
        for q in self.q_vectors():
            if q == -q:
                yield q

    def _gen_B(self):
        b = []
        for q in self.q_vectors():
            if q != -q and -q not in b:
                b.append(q)
                yield q

    def operator_q_vectors(self):
        return it.chain(self._gen_A(), self._gen_B())

    def _ops(self, kappa, alpha, p, op):
        dimx = self.dim_space * self.M * self.Np
        xop = [qutip.qeye(self._nfock)] * dimx
        dims = it.chain([self.dim_space, self.M], self.N)
        coefs = it.chain([alpha, kappa], p)
        idx = 0
        for xi, ni in zip(coefs, it.accumulate(dims, func=operator.mul)):
            idx += xi * ni
        xop.insert(idx, op)
        xop.pop(idx+1)
        return qutip.tensor(xop)

    def _x_ops(self, kappa, alpha, p):
        return self._ops(kappa, alpha, p, op=qutip.position(self._nfock))

    def _p_ops(self, kappa, alpha, p):
        return self._ops(kappa, alpha, p, op=qutip.momentum(self._nfock))

    def _z(self, q, v):
        real_z = 0+0j
        imag_z = 0+0j
        real_pz = 0+0j
        imag_pz = 0+0j
        for kappa, alpha, p in it.product(range(self.M), range(self.dim_space),
                                          self.unit_cells()):
            zi = exp(-1j * 2*np.pi * (q.dot(p)))
            zi *= sqrt(self.unit_cell.mass(kappa) / self.unit_cell.mass(0))
            zi *= np.conj(self.e(kappa, alpha, v)(q)) / sqrt(self.Np)
            xop = self._x_ops(kappa, alpha, p)
            pop = self._p_ops(kappa, alpha, p)
            real_z += zi.real * xop
            imag_z += zi.imag * xop
            real_pz += zi.real * pop
            imag_pz += zi.imag * pop
        return real_z, imag_z, real_pz, imag_pz

    def _l_inv(self, q, v):
        m0 = self.unit_cell.mass(0)
        # omega2 = self.omega2(q, v)+0j
        omega2 = 1
        return sqrt(2 * m0 * sqrt(omega2))

    def _x(self, q, v):
        real_z, imag_z, real_pz, imag_pz = self._z(q, v)
        linv = self._l_inv(q, v)
        if q in self.A:
            return real_z/2 * linv, real_pz/2 * linv
        elif q in self.B:
            return real_z * linv, real_pz * linv

    def _y(self, q, v):
        real_z, imag_z, real_pz, imag_pz = self._z(q, v)
        linv = self._l_inv(q, v)
        if q in self.B:
            return imag_z * linv, imag_pz * linv

    def _get_matrix_rep_d(self, q):
        d_mat = np.empty(shape=(self.num_modes, self.num_modes), dtype=np.complex)
        for k1x1, i in zip(self._particles(), it.count()):
            k1, x1 = k1x1
            for k2x2, j in zip(self._particles(), it.count()):
                k2, x2 = k2x2
                d_mat[i, j] = self.d_matrix(
                    kappa1=k1, alpha1=x1, kappa2=k2, alpha2=x2)(q)
        return d_mat

    def _orthonormal_eigenvectors(self, dmat):
        evals, evects = eigh(a=dmat, turbo=True)
        on_evects = dot(evects, inv(sqrtm(dot(evects.conj().T, evects))))
        return evals, [on_evects[:, i] for i in range(len(evals))]

    def _set_d_eigenvectors(self, q):
        dmat = self._get_matrix_rep_d(q)
        evals, evects = self._orthonormal_eigenvectors(dmat=dmat)
        self._evals_dict[q] = evals
        self._evect_dict[q] = evects


class PhononLattice1D(PhononLattice):
    def __init__(self, unit_cell, N_x, c_matrix, n_fock=2, *args, **kwargs):
        super(PhononLattice1D, self).__init__(
            unit_cell=unit_cell, N=[N_x], c_matrix=c_matrix, nfock=n_fock,
            *args, **kwargs)
        self.Nx = N_x


class PhononLattice2D(PhononLattice):
    def __init__(self, unit_cell, N_x, N_y, c_matrix, n_fock=2, *args,
                 **kwargs):
        super(PhononLattice2D, self).__init__(
            unit_cell=unit_cell, N=[N_x, N_y], c_matrix=c_matrix, nfock=n_fock,
            *args, **kwargs)
        self.Nx = N_x
        self.Ny = N_y


class PhononLattice3D(PhononLattice):
    def __init__(self, unit_cell, N_x, N_y, N_z, c_matrix, n_fock=2,
                 *args, **kwargs):
        super(PhononLattice3D, self).__init__(
            unit_cell=unit_cell, N=[N_x, N_y, N_z], c_matrix=c_matrix,
            nfock=n_fock, *args, **kwargs)
        self.Nx = N_x
        self.Ny = N_y
        self.Nz = N_z


def get_c_matrix_simple_harmonic_interaction(k):
    def c_matrix(lattice, kappa1, alpha1, p1, kappa2, alpha2, p2):
        if (kappa1, alpha1, p1.all()) == (kappa2, alpha2, p2.all()):
            return k * lattice.unit_cell.num_connections(kappa1)
        elif alpha1 == alpha2 and lattice.are_connected(
                kappa1=kappa1, p1=p1, kappa2=kappa2, p2=p2):
            return -k
        else:
            return 0
    return c_matrix


def get_c_matrix_coulomb_interaction(g):
    def c_matrix(lattice, kappa1, alpha1, p1, kappa2, alpha2, p2):
        if not lattice.are_connected(kappa1, p1, kappa2, p2):
            return 0  # Only interact with neighbors

        def sterm(kappa_i, p_i):
            if (kappa1, p1) == (kappa_i, p_i):
                return 0
            disp = lattice.periodic_displacement_distance(
                kappa1=kappa_i, p1=p_i, kappa2=kappa1, p2=p1
            ).displacement()
            t1 = -g * 3 * disp[alpha1] * disp[alpha2] / norm(disp, ord=2)**5
            if alpha1 != alpha2:
                return t1
            else:
                return t1 + g / norm(disp, ord=2)**3

        if kappa1 == kappa2:
            s = 0
            for kappa_i, p_i in it.product(range(lattice.M), lattice.unit_cells()):
                s += sterm(kappa_i, p_i)
            return s
        else:
            return -sterm(kappa2, p2)
    return c_matrix


def _taudisp(lattice, tau_ind):
    alpha, k1, p1, k2, p2 = tau_ind
    disp = lattice.periodic_displacement_distance(k1, p1, k2, p2)
    return disp.displacement()[alpha]


def _delta(lattice, delta_ind):
    kappa1, p1, kappa2, p2 = delta_ind
    if (kappa1, p1) == (kappa2, p2):
        return 0
    else:
        dtau = lattice.periodic_displacement_distance(kappa1, p1, kappa2, p2)
        return 1/norm(dtau, ord=2)


def _d_delta(lattice, d_ind, delta_ind):
    k, alpha, p = d_ind
    k1, p1, k2, p2 = delta_ind
    if (k, p) != (k1, p1) and (k, p) != (k2, p2):
        return 0
    taudisp = _taudisp(lattice, [alpha, k1, p1, k2, p2])
    delta = _delta(lattice, delta_ind)
    d = -taudisp * delta**3
    if (k, p) == (k1, p1):
        return d
    elif (k, p) == (k2, p2):
        return -d
    else:
        return 0


def _d_taudisp(lattice, d_ind, tau_ind):
    kd, alphad, pd = d_ind
    alpha, k1, p1, k2, p2 = tau_ind
    if (kd, alphad, pd) == (k1, alpha, p1):
        return 1
    elif (kd, alphad, pd) == (k2, alpha, p2):
        return -1
    else:
        return 0


def _d_d_delta(lattice, d_ind_a, d_ind_b, delta_ind):
    k_a, alpha_a, p_a = d_ind_a
    k1, p1, k2, p2 = delta_ind
    if (k_a, p_a) != (k1, p1) and (k_a, p_a) != (k2, p2):
        return 0
    delta = _delta(lattice, delta_ind=delta_ind)
    ddelta = _d_delta(lattice, d_ind=d_ind_b, delta_ind=delta_ind)
    tau_ind = [alpha_a, k1, p1, k2, p2]
    taudisp = _taudisp(lattice, tau_ind=tau_ind)
    dtau = -_d_taudisp(lattice, d_ind=d_ind_b, tau_ind=tau_ind)
    dddelta = -dtau * delta**3 - 3 * taudisp * ddelta * delta**2
    if (k_a, p_a) == (k1, p1):
        return dddelta
    elif (k_a, p_a) == (k2, p2):
        return -dddelta
    else:
        return 0


def get_c_matrix_coulomb_interaction2(k):
    def filterfn(iterable):
        return filter(lambda x: x[0] <= x[1], iterable)

    def cmat(lattice, kappa1, alpha1, p1, kappa2, alpha2, p2):
        charge = lattice.unit_cell.charge
        c = 0
        for kpair, ppair in it.product(
                filterfn(it.product(range(lattice.M), repeat=2)),
                filterfn(it.product(lattice.unit_cells(), repeat=2)),
        ):
            ka, kb = kpair
            pa, pb = ppair
            c += k * charge(ka) * charge(kb) * _d_d_delta(
                lattice,
                d_ind_a=[kappa1, alpha1, p1],
                d_ind_b=[kappa2, alpha2, p2],
                delta_ind=[ka, pa, kb, pb]
            )
        return c

    return cmat
