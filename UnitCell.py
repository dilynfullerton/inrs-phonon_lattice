import itertools as it

import numpy as np
from numpy import array, dot, cos, sin
from scipy.linalg import det


def _all_connections(connections):
    sorted_connections = set()
    for k1, k2 in connections:
        sorted_connections.add((k2, k1))
        sorted_connections.add((k1, k2))
    return sorted_connections


class CellInformationNotProvidedError(RuntimeError):
    pass


class UnitCell:
    def __init__(
            self, a_vectors, particle_positions, particle_masses, connections,
            particle_charges=None
    ):
        self.a_vectors = [array(a) for a in a_vectors]
        self.a_matrix = np.vstack(self.a_vectors).T
        self.dim = len(a_vectors)
        self.particle_positions = [array(pos) for pos in particle_positions]
        self.particle_masses = particle_masses
        self.particle_charges = particle_charges
        self.connections = connections
        self.num_particles = len(self.particle_positions)

    def num_connections(self, i):
        """Returns the number of connections to the ith particle in the cell
        """
        nc = 0
        for ec_arr in self.connections:
            for ec in ec_arr:
                if ec[0] == i:
                    nc += 1
                if ec[1] == i:
                    nc += 1
        return nc

    def connected(self, i, j, disp):
        """Returns True if particle i in this unit cell is connected to
        particle j in the adjacent unit cell displaced by disp
        :param i: index of particle in this cell
        :param j: index of particle in neighboring cell
        :param disp: Vector of (0s and 1s) or (0s and -1s) specifying the
        relative displacement in each dimension of the neighboring cell
        containing j
        """
        disp = disp.canonical_displacement()
        idx = sum([2 ** i * a for a, i in zip(reversed(disp), it.count())])
        if idx < 0:
            idx = -idx
            i, j = j, i
        c = self.connections[idx]
        if idx == 0:
            return (i, j) in c or (j, i) in c
        else:
            return (i, j) in c

    def connected_int(self, i, j):
        """Returns True if particle i is connected to particle j in this cell;
        otherwise False
        """
        return self.connected(i, j, disp=[0] * self.dim)

    def mass(self, i):
        """Returns the mass of particle i in this cell
        """
        return self.particle_masses[i]

    def charge(self, i):
        if self.particle_charges is None:
            raise CellInformationNotProvidedError(
                'Particle charge data was not provided in unit cell'
                ' definition'
            )
        else:
            return self.particle_charges[i]

    def particle_displacement_in_cell_mod_a(self, i, j):
        """Returns the displacement of particle i from particle j with
        respect to the ordered basis self.a_vectors
        """
        return self.particle_positions[i] - self.particle_positions[j]

    def particle_displacement_distance_in_cell(self, k1, k2):
        """Returns the absolute displacement vector of particle i from
        particle j
        """
        return np.dot(self.a_matrix,
                      self.particle_displacement_in_cell_mod_a(k1, k2))


class UnitCell1D(UnitCell):
    def __init__(
            self, a1,
            particle_positions,
            particle_masses,
            internal_connections=list(),
            external_connections_x=list(),
            *args, **kwargs
    ):
        """Creates a 1D unit cell with the given edge vector a1,
        particle positions, and particle connections.

        a1 is a position-space vectors in R^a defining the edge of
        the cell. Namely, the cell be defined by two corners:
            {0, a1}

        The cell will contain N particles, where N = len(particle_positions).
        The particles will be assigned labels
            0, 1, 2, ..., N-1
        according to their positions in the list

        The values of the list particle_positions are one-vectors
        describing the associated particle's position in the cell relative
        to the basis vector a1. Namely, if particle j has position
        [xj], then this is interpreted as
            xj * a1

        The connections between particles are given by internal_connections
        and external_connections_x

        internal_connections is a list of 2-tuples (p1, p2) of particle
        labels which are given a connection.

        external_connections_x is also a list of 2-tuples
        of particle labels, but in these the first label refers to a
        particle in this cell, while the second label refers to a particle
        in the cell to the right

        :param a1: one-vector specifying the edge of the unit cell
        :param particle_positions: list of one-vectors [xj] in
        the ordered basis {a1}, specifying the positions of the
        particles in the cell. Note these must satisfy all of
            0 <= xj < 1
        This is to ensure that arraying the cells in a lattice does not
        result in particles on top of each other.
        :param internal_connections: List of 2-tuples (p1, p2) where p1 and
        p2 are distinct particle labels.
        :param external_connections_x: List of 2-tuples (p1, p2) where p1 and
        p2 are particle labels. Here p1 refers to the particle in
        this cell and p2 refers to the particle in the adjacent cell to the
        right.
        """
        super(UnitCell1D, self).__init__(
            a_vectors=[a1],
            particle_positions=particle_positions,
            particle_masses=particle_masses,
            connections=[
                internal_connections,  # 0 = x
                external_connections_x,  # 1
            ],
            *args, **kwargs
        )
        self.a1 = self.a_vectors[0]

    def connected_x(self, k1, k2):
        return self.connected(k1, k2, disp=1)


class UnitCell2D(UnitCell):
    def __init__(
            self, a1, a2,
            particle_positions,
            particle_masses,
            internal_connections=list(),
            external_connections_x=list(),
            external_connections_y=list(),
            external_connections_xy=list(),
            *args, **kwargs
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
        super(UnitCell2D, self).__init__(
            a_vectors=[a1, a2],
            particle_positions=particle_positions,
            particle_masses=particle_masses,
            connections=[
                internal_connections,  # 00 = xy
                external_connections_y,  # 01
                external_connections_x,  # 10
                external_connections_xy,  # 11
            ],
            *args, **kwargs
        )
        self.a1 = self.a_vectors[0]
        self.a2 = self.a_vectors[1]

    def connected_y(self, k1, k2):
        return self.connected(k1, k2, disp=(0, 1))

    def connected_x(self, k1, k2):
        return self.connected(k1, k2, disp=(1, 0))

    def connected_xy(self, k1, k2):
        return self.connected(k1, k2, disp=(1, 1))


class UnitCell3D(UnitCell):
    def __init__(
            self, a1, a2, a3,
            particle_positions,
            particle_masses,
            internal_connections=list(),
            external_connections_x=list(),
            external_connections_y=list(),
            external_connections_z=list(),
            external_connections_xy=list(),
            external_connections_xz=list(),
            external_connections_yz=list(),
            external_connections_xyz=list(),
            *args, **kwargs
    ):
        """Creates a 3D unit cell with the given edge vectors a1, a2, adn a3,
        particle positions, and particle connections.

        a1, a2, and a3 are position-space vectors in R^3 defining the
        bottom-left-front corner of the parallelopied. Namely, the cell be
        defined by the six points
            {0, a1, a2, a3, a1+a2, a2+a3, a3+a1, a1+a2+a3}

        The cell will contain N particles, where N = len(particle_positions).
        The particles will be assigned labels
            0, 1, 2, ..., N-1
        according to their positions in the list

        The values of the list particle_positions are three-vectors
        describing the associated particle's position in the cell relative
        to the basis vectors {a1, a2, a3}. Namely, if particle j has position
        (xj, yj, zj), then this is interpreted as
            xj * a1 + yj * a2 + zj * a3

        The connections between particles are given by internal_connections,
        external_connections_x, external_connections_y, and
        external_connections_z.

        internal_connections is a list of 2-tuples (p1, p2) of particle
        labels which are given a connection.

        external_connections_x, external_connections_y, and
        external_connections_z are also 2-tuples
        of particle labels, but in these the first label refers to a
        particle in this cell, while the second label refers to a particle
        in the adjacent cell in the direction +x if _x, +y if _y, +z if _z.

        :param a1: 3-vector specifying the bottom-front edge of the cell
        :param a2: 3-vector specifying the bottom-left edge of the cell
        :param a3: 3-vector specifying the front-left edge of the cell
        :param particle_positions: list of 3-vectors (xj, yj, zj) in
        the ordered basis {a1, a2, a3}, specifying the positions of the
        particles in the cell. Note these must satisfy all of
            0 <= xj < 1
            0 <= yj < 1
            0 <= zj < 1
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
        this cell and p2 refers to the particle in the adjacent cell behind
        :param external_connections_z: List of 2-tuples (p1, p2) where p1 and
        p2 are particle labels. Here p1 refers to the particle in
        this cell and p2 refers to the particle in the adjacent cell above.
        """
        super(UnitCell3D, self).__init__(
            a_vectors=[a1, a2, a3],
            particle_positions=particle_positions,
            particle_masses=particle_masses,
            connections=[
                internal_connections,      # 000 = xyz
                external_connections_z,    # 001
                external_connections_y,    # 010
                external_connections_yz,   # 011
                external_connections_x,    # 100
                external_connections_xz,   # 101
                external_connections_xy,   # 110
                external_connections_xyz,  # 111
            ],
            *args,
            **kwargs
        )
        self.a1 = self.a_vectors[0]
        self.a2 = self.a_vectors[1]
        self.a3 = self.a_vectors[2]


# Functions for easy generation
def _unit_a_matrix(dim, angles):
    amat = np.eye(dim)
    for theta_ij, ij in zip(angles, it.combinations(range(dim), r=2)):
        i, j = ij
        tij = np.eye(dim)
        tij[i, j] = np.tan(theta_ij)
        amat = np.dot(tij, amat)
    return amat


def _unit_a_vectors(dim, angles):
    amat = _unit_a_matrix(dim, angles)
    return [amat[:, i] for i in range(dim)]


def a_from_density(density, total_mass, unit_amat):
    unit_vol = det(unit_amat)
    a3 = total_mass / density / unit_vol
    a = a3**(1/3)
    return a


def _make_unit_cell(a, cell_type, angles, dim, masses, a_is_density=False,
                    *args, **kwargs):
    amat = _unit_a_matrix(dim, angles)
    if a_is_density:
        a = a_from_density(density=a, total_mass=sum(masses), unit_amat=amat)
    a_vectors = [a * amat[:, i] for i in range(len(dim))]
    positions, connections = cell_type
    return UnitCell(
        a_vectors=a_vectors,
        particle_positions=positions,
        particle_masses=masses,
        connections=connections,
        *args, **kwargs
    )


def simple_cubic3d(a, mass, phi=0, theta=0, psi=0, a_is_density=False,
                   *args, **kwargs):
    particle_masses = [mass]
    unit_vectors = _unit_a_vectors(dim=3, angles=[phi, theta, psi])
    if a_is_density:
        a = a_from_density(
            density=a, total_mass=sum(particle_masses),
            unit_amat=np.column_stack(unit_vectors)
        )
    return UnitCell3D(
        a1=a * unit_vectors[0],
        a2=a * unit_vectors[1],
        a3=a * unit_vectors[2],
        particle_positions=[[0, 0, 0]],
        particle_masses=particle_masses,
        internal_connections=[],
        external_connections_x=[(0, 0)],
        external_connections_y=[(0, 0)],
        external_connections_z=[(0, 0)],
        *args, **kwargs
    )


def body_centered_cubic3d(a, mass_corner, mass_center, phi=0, theta=0, psi=0,
                          a_is_density=False, *args, **kwargs):
    particle_masses = [mass_corner, mass_center]
    unit_vectors = _unit_a_vectors(dim=3, angles=[phi, theta, psi])
    if a_is_density:
        a = a_from_density(
            density=a, total_mass=sum(particle_masses),
            unit_amat=np.column_stack(unit_vectors)
        )
    return UnitCell3D(
        a1=a * unit_vectors[0],
        a2=a * unit_vectors[1],
        a3=a * unit_vectors[2],
        particle_masses=particle_masses,
        particle_positions=[[0, 0, 0], [1/2, 1/2, 1/2]],
        internal_connections=[(0, 1)],
        external_connections_x=[(0, 0), (1, 0)],
        external_connections_y=[(0, 0), (1, 0)],
        external_connections_z=[(0, 0), (1, 0)],
        external_connections_xy=[(1, 0)],
        external_connections_yz=[(1, 0)],
        external_connections_xz=[(1, 0)],
        external_connections_xyz=[(1, 0)],
        *args, **kwargs
    )


def face_centered_cubic3d(a, mass_corner, mass_face, phi=0, theta=0, psi=0,
                          a_is_density=False, *args, **kwargs):
    particle_masses = [mass_corner, mass_face, mass_face, mass_face]
    unit_vectors = _unit_a_vectors(dim=3, angles=[phi, theta, psi])
    if a_is_density:
        a = a_from_density(
            density=a, total_mass=sum(particle_masses),
            unit_amat=np.column_stack(unit_vectors)
        )
    return UnitCell3D(
        a1=a * unit_vectors[0],
        a2=a * unit_vectors[1],
        a3=a * unit_vectors[2],
        particle_masses=particle_masses,
        particle_positions=[
            [0, 0, 0],  # corner
            [0, 1/2, 1/2], [1/2, 0, 1/2], [1/2, 1/2, 0],  # faces
        ],
        internal_connections=[(0, 1), (0, 2), (0, 3)],
        external_connections_x=[(0, 0), (2, 0), (3, 0)],
        external_connections_y=[(0, 0), (3, 0), (1, 0)],
        external_connections_z=[(0, 0), (1, 0), (2, 0)],
        external_connections_xy=[(1, 0), (3, 0)],
        external_connections_yz=[(1, 0), (1, 0)],
        external_connections_xz=[(1, 0), (2, 0)],
        external_connections_xyz=[],
        *args, **kwargs,
    )


def base_centered_cubic3d(a, mass_corner, mass_face, phi=0, theta=0, psi=0,
                          a_is_density=False, *args, **kwargs):
    particle_masses = [mass_corner, mass_face]
    unit_vectors = _unit_a_vectors(dim=3, angles=[phi, theta, psi])
    if a_is_density:
        a = a_from_density(
            density=a, total_mass=sum(particle_masses),
            unit_amat=np.column_stack(unit_vectors)
        )
    return UnitCell3D(
        a1=a * unit_vectors[0],
        a2=a * unit_vectors[1],
        a3=a * unit_vectors[2],
        particle_masses=particle_masses,
        particle_positions=[
            [0, 0, 0],  # corner
            [1/2, 1/2, 0],  # face
        ],
        internal_connections=[(0, 1)],
        external_connections_x=[(0, 0), (1, 0)],
        external_connections_y=[(0, 0), (1, 0)],
        external_connections_z=[(0, 0)],
        external_connections_xy=[(1, 0)],
        external_connections_yz=[],
        external_connections_xz=[],
        external_connections_xyz=[],
        *args, **kwargs,
    )


def zincblende3d(a, mass1, mass2, charge1, charge2, phi=0, theta=0, psi=0,
                 a_is_density=False, *args, **kwargs):
    particle_masses = [
        mass1, mass1, mass1, mass1, mass2, mass2, mass2, mass2
    ]
    particle_charges = [
        charge1, charge1, charge1, charge1, charge2, charge2, charge2, charge2
    ]
    unit_vectors = _unit_a_vectors(dim=3, angles=[phi, theta, psi])
    if a_is_density:
        a = a_from_density(
            density=a, total_mass=sum(particle_masses),
            unit_amat=np.column_stack(unit_vectors)
        )
    return UnitCell3D(
        a1=a * unit_vectors[0],
        a2=a * unit_vectors[1],
        a3=a * unit_vectors[2],
        particle_masses=particle_masses,
        particle_charges=particle_charges,
        particle_positions=[
            [0, 0, 0], [1/2, 1/2, 0], [1/2, 0, 1/2], [0, 1/2, 1/2],
            [1/4, 1/4, 1/4], [3/4, 3/4, 1/4], [3/4, 1/4, 3/4], [1/4, 3/4, 3/4],
        ],
        internal_connections=[
            (0, 4), (1, 4), (2, 4), (3, 4),
            (1, 5), (2, 6), (3, 7)
        ],
        external_connections_x=[(5, 3), (6, 3)],
        external_connections_y=[(7, 2), (5, 2)],
        external_connections_z=[(6, 1), (7, 1)],
        external_connections_xy=[(5, 0)],
        external_connections_yz=[(7, 0)],
        external_connections_xz=[(6, 0)],
        external_connections_xyz=[],
        *args, **kwargs,
    )
