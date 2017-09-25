from itertools import product, count, chain
from numpy import exp, dot, array, sqrt
from scipy.linalg import sqrtm, eigh, inv
import numpy as np
from qutip import *


def _bloch_equiv(q1, q2):
    for q1x, q1y in zip(q1, q2):
        diff = q1x - q1y
        if diff != int(diff):
            return False
    else:
        return True


def _all_connections(connections):
    sorted_connections = set()
    for k1, k2 in connections:
        sorted_connections.add((k2, k1))
        sorted_connections.add((k1, k2))
    return sorted_connections


class UnitCell2D:
    def __init__(
            self,
            a1, a2,
            particle_positions,
            particle_masses,
            internal_connections,
            external_connections_x,
            external_connections_y,
    ):
        """Creates a 2D unit cell with the given edge vectors a1 and a2,
        particle positions, and particle connections.

        a1 and a2 are position-space vectors in R^2 defining the left and
        bottom edges of the cell. Namely, the cell be defined by the four
        points:
            {0, a1, a2, a1+a2}

        The cell will contain N particles, where N = len(particle_positions).
        The particles will be assigned labels
            0, 1, 2, ..., N-1
        according to their positions in the list

        The values of the list particle_positions are two-vectors
        describing the associated particle's position in the cell relative
        to the basis vectors b1 and b2. Namely, if particle j has position
        (xj, yj), then this is interpreted as
            xj * a1 + yj * a2

        The connections between particles are given by internal_connections,
        external_connections_x, and external_connections_y.

        internal_connections is a list of 2-tuples (p1, p2) of particle
        labels which are given a connection.

        external_connections_x and external_connections_y are also 2-tuples
        of particle labels, but in these the first label refers to a
        particle in this cell, while the second label refers to a particle
        in the cell to the right if _x or above if _y.

        :param b1: 2-vector specifying the lower edge of the unit cell
        :param b2: 2-vector specifying the left edge of the unit cell
        :param particle_positions: list of 2-vectors (xj, yj) in
        the ordered basis {b1, b2}, specifying the positions of the
        particles in the cell. Note these must satisfy all of
            0 <= xj < 1
            0 <= yj < 1
        This is to ensure that arraying the cells in a lattice does not
        result in particles on top of each other.
        :param internal_connections: List of 2-tuples (p1, p2) where p1 and
        p2 are distinct particle labels.
        :param external_connections_x: List of 2-tuples (p1, p2) where p1 and
        p2 are particle labels. Here p1 refers to the particle in
        this cell and p2 refers to the particle in the adjacent cell to the
        right.
        :param external_connections_y: List of 2-tuples (p1, p2) where p1 and
        p2 are particle labels. Here p1 refers to the particle in
        this cell and p2 refers to the particle in the adjacent cell above.
        """
        self.a1 = array(a1)
        self.a2 = array(a2)
        self.particle_positions = particle_positions
        self.particle_masses = particle_masses
        self.internal_connections = _all_connections(internal_connections)
        self.external_connections_x = external_connections_x
        self.external_connections_y = external_connections_y
        self.num_particles = len(self.particle_positions)

    def connected_int(self, k1, k2):
        return (k1, k2) in self.internal_connections

    def connected_x(self, k1, k2):
        return (k1, k2) in self.external_connections_x

    def connected_y(self, k1, k2):
        return (k1, k2) in self.external_connections_y

    def mass(self, k):
        return self.particle_masses[k]

    def position(self, k, p=(0, 0)):
        x, y = self.particle_positions[k]
        return (x + p[0]) * self.a1 + (y + p[1]) * self.a2


# Examples
LINE_2D = UnitCell2D(
    a1=(1, 0), a2=(0, 1),
    particle_positions=[(0, 0)],
    particle_masses=[0],
    internal_connections=[],
    external_connections_x=[(0, 0)],
    external_connections_y=[]
)
SQUARE_2D = UnitCell2D(
    a1=(1, 0), a2=(0, 1),
    particle_positions=[(0, 0)],
    particle_masses=[0],
    internal_connections=[],
    external_connections_x=[(0, 0)],
    external_connections_y=[(0, 0)]
)


class BlochVector:
    def __init__(self, q):
        self.q = array(q)

    def __eq__(self, other):
        if not isinstance(other, BlochVector):
            return False
        else:
            return self.canonical_form().all() == other.canonical_form().all()

    def __hash__(self):
        return int(dot(self.q, self.q))

    def __neg__(self):
        return BlochVector(-self.q)

    def __add__(self, other):
        return BlochVector(self.q + other.q)

    def __sub__(self, other):
        return BlochVector(self.q - other.q)

    def __mul__(self, other):
        return BlochVector(other * self.q)

    def __getitem__(self, item):
        return self.q[item]

    def __len__(self):
        return len(self.q)

    def canonical_form(self):
        return array([qi - int(qi) for qi in self.q])

    def dot(self, other):
        assert len(other) == len(self)
        other = array(other)
        return np.dot(self.q, other)


class Lattice2D:
    def __init__(self, unit_cell, N_x, N_y, c_matrix):
        self.unit_cell = unit_cell
        self.Nx = N_x
        self.Ny = N_y
        self.M = self.unit_cell.num_particles
        # self.b1, self.b2 = self._set_b_vectors()
        self.c_matrix = c_matrix
        self._evals_dict = dict()
        self._evect_dict = dict()
        self._indices = list(
            product(range(self.M), range(2))
        )

    def are_connected(self, k1, p1, k2, p2):
        """Returns true if there is a connection between particle number k1
        in unit cell p1 and particle number k2 in unit cell p2
        :param k1: Index of particle 1 in unit cell
        :param p1: Position index (nx, ny) of unit cell 1
        :param k2: Index of particle 2 in unit cell
        :param p2: Position index (nx, ny) of unit cell 2
        """
        nx1, ny1 = p1
        nx2, ny2 = p2
        if abs(nx1-nx2) + abs(ny1-ny2) > 1:
            return False
        elif nx1 - nx2 == 1 or nx2 - nx1 == self.Nx - 1:
            return self.unit_cell.connected_x(k2, k1)
        elif nx2 - nx1 == 1 or nx1 - nx2 == self.Nx - 1:
            return self.unit_cell.connected_x(k1, k2)
        elif ny1 - ny2 == 1 or ny2 - ny1 == self.Ny - 1:
            return self.unit_cell.connected_y(k2, k1)
        elif ny2 - ny1 == 1 or ny1 - ny2 == self.Ny - 1:
            return self.unit_cell.connected_y(k1, k2)
        else:
            return self.unit_cell.connected_int(k1, k2)

    def position(self, k, p):
        return self.unit_cell.position(k, p)

    def displacement(self, k1, p1, k2, p2):
        return self.position(k2, p2) - self.position(k1, p1)

    def unit_cells(self):
        return product(range(self.Nx), range(self.Ny))

    def d_matrix(self, k1, x1, k2, x2):
        def _d(q):
            d = 0
            for p in self.unit_cells():
                dp = self.c_matrix(self, k1, x1, p, k2, x2, p)
                dp *= exp(1j * 2*np.pi * q.dot(p))
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
        for m1, m2 in self.unit_cells():
            yield BlochVector([m1/self.Nx, m2/self.Ny])

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
        return chain(self._A(), self._B())

    def annihilation_operator(self, q, v):
        if q in self._A():
            x, px = self._x(q, v)
            return 1/sqrt(2) * (x + 1j*px)
        elif q in self._B():
            x, px = self._x(q, v)
            y, py = self._y(q, v)
            return 1/2 * (x + 1j*px) + 1j/2 * (y + 1j*py)

    def _ops(self, k, x, p, op):
        dim = 2 * self.M * self.Nx * self.Ny
        nx, ny = p
        xop = [qeye(2)] * dim
        idx = x + 2*k + 2*self.M*nx + 2*self.M*self.Nx*ny
        xop.insert(idx, op)
        xop.pop(idx+1)
        return tensor(xop)

    def _x_ops(self, k, x, p):
        return self._ops(k, x, p, op=position(2))

    def _p_ops(self, k, x, p):
        return self._ops(k, x, p, op=momentum(2))

    def _z(self, q, v):
        real_z = 0+0j
        imag_z = 0+0j
        real_pz = 0+0j
        imag_pz = 0+0j
        for k, x, p in product(range(self.M), range(2), self.unit_cells()):
            zi = exp(-1j * 2*np.pi * (q[0]*p[0] + q[1]*p[1]))
            zi *= sqrt(self.unit_cell.mass(k)/self.unit_cell.mass(0))
            zi *= np.conj(self.e(k, x, v)(q)) / sqrt(self.Nx * self.Ny)
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

    # def _set_b_vectors(self):
    #     a1, a2 = self.unit_cell.a1, self.unit_cell.a2
    #     amat = np.vstack((a1, a2))
    #     bmat = inv(amat) * 2*np.pi
    #     return bmat[:, 0], bmat[:, 1]

    def _get_matrix_rep_d(self, q):
        nparts = self.unit_cell.num_particles
        dim = nparts * 2
        d_mat = np.empty(shape=(dim, dim), dtype=np.complex)
        for k1x1, i in zip(product(range(nparts), range(2)), count()):
            k1, x1 = k1x1
            for k2x2, j in zip(product(range(k1, nparts), range(x1, 2)),
                               count()):
                k2, x2 = k2x2
                d_mat[i, j] = self.d_matrix(k1, x1, k2, x2)(q)
                if i != j:
                    d_mat[j, i] = np.conj(d_mat[i][j])
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


def get_cmat_harmonic_interaction2d(k, g):
    def cmat_harmonic_interaction2d(lattice2d, k1, x1, p1, k2, x2, p2):
        if (k1, x1, p1) == (k2, x2, p2):
            return k
        elif lattice2d.are_connected(k1, p1, k2, p2):
            return -g/2
        return 0
    return cmat_harmonic_interaction2d


# Simple example
line_x = UnitCell2D(
    a1=[1, 0], a2=[0, 1],
    particle_positions=[0],
    particle_masses=[1],
    internal_connections=[],
    external_connections_x=[(0, 0)],
    external_connections_y=[(0, 0)]
)

c_mat = get_cmat_harmonic_interaction2d(k=1, g=1/100)

lattice = Lattice2D(
    unit_cell=line_x, N_x=2, N_y=2, c_matrix=c_mat,
)

for q in lattice.operator_q_vectors():
    for v in range(lattice.M * 2):
        a = lattice.annihilation_operator(q, v)
        print(a)
