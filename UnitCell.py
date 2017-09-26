import itertools as it

import numpy as np
from numpy import array, dot, cos, sin


def _all_connections(connections):
    sorted_connections = set()
    for k1, k2 in connections:
        sorted_connections.add((k2, k1))
        sorted_connections.add((k1, k2))
    return sorted_connections


class UnitCell:
    def __init__(
            self,
            a_vectors,
            particle_positions,
            particle_masses,
            connections,
    ):
        self.a_vectors = [array(a) for a in a_vectors]
        self.a_matrix = np.vstack(self.a_vectors).T
        self.dim = len(a_vectors)
        self.particle_positions = [array(pos) for pos in particle_positions]
        self.particle_masses = particle_masses
        self.connections = connections
        self.num_particles = len(self.particle_positions)

    def num_connections(self, k):
        nc = 0
        for ec_arr in self.connections:
            for ec in ec_arr:
                if ec[0] == k:
                    nc += 1
                if ec[1] == k:
                    nc += 1
        return nc

    def connected(self, k1, k2, axis):
        idx = sum([2**i * a for a, i in zip(reversed(axis), it.count())])
        if idx < 0:
            idx = -idx
            k1, k2 = k2, k1
        c = self.connections[idx]
        if idx == 0:
            return (k1, k2) in c or (k2, k1) in c
        else:
            return (k1, k2) in c

    def connected_int(self, k1, k2):
        return self.connected(k1, k2, axis=[0]*self.dim)

    def mass(self, k):
        return self.particle_masses[k]

    def displacement_mod_a(self, k1, k2):
        return self.particle_positions[k1] - self.particle_positions[k2]

    def displacement(self, k1, k2):
        return dot(self.a_matrix, self.displacement_mod_a(k1, k2))


class UnitCell1D(UnitCell):
    def __init__(
            self, a1,
            particle_positions,
            particle_masses,
            internal_connections=list(),
            external_connections_x=list(),
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
            ]
        )
        self.a1 = self.a_vectors[0]

    def connected_x(self, k1, k2):
        return self.connected(k1, k2, axis=1)


class UnitCell2D(UnitCell):
    def __init__(
            self, a1, a2,
            particle_positions,
            particle_masses,
            internal_connections=list(),
            external_connections_x=list(),
            external_connections_y=list(),
            external_connections_xy=list(),
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
            ]
        )
        self.a1 = self.a_vectors[0]
        self.a2 = self.a_vectors[1]

    def connected_y(self, k1, k2):
        return self.connected(k1, k2, axis=(0, 1))

    def connected_x(self, k1, k2):
        return self.connected(k1, k2, axis=(1, 0))

    def connected_xy(self, k1, k2):
        return self.connected(k1, k2, axis=(1, 1))


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
            external_connections_xyz=list()
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
            ]
        )
        self.a1 = self.a_vectors[0]
        self.a2 = self.a_vectors[1]
        self.a3 = self.a_vectors[2]


# Functions for easy generation
def simple_cubic3d(a, mass, phi=0, theta=0, psi=0):
    return UnitCell3D(
        a1=a * array([1, 0, 0]),
        a2=a * array([sin(phi), cos(phi), 0]),
        a3=a * array([sin(theta)*cos(psi), sin(theta)*sin(psi), cos(theta)]),
        particle_positions=[[0, 0, 0]],
        particle_masses=[mass],
        internal_connections=[],
        external_connections_x=[(0, 0)],
        external_connections_y=[(0, 0)],
        external_connections_z=[(0, 0)]
    )


def body_centered_cubic3d(a, mass_corner, mass_center, phi=0, theta=0, psi=0):
    return UnitCell3D(
        a1=a * array([1, 0, 0]),
        a2=a * array([sin(phi), cos(phi), 0]),
        a3=a * array([sin(theta)*cos(psi), sin(theta)*sin(psi), cos(theta)]),
        particle_positions=[[0, 0, 0], [1/2, 1/2, 1/2]],
        particle_masses=[mass_corner, mass_center],
        internal_connections=[(0, 1)],
        external_connections_x=[(0, 0), (1, 0)],
        external_connections_y=[(0, 0), (1, 0)],
        external_connections_z=[(0, 0), (1, 0)],
        external_connections_xy=[(1, 0)],
        external_connections_yz=[(1, 0)],
        external_connections_xz=[(1, 0)],
        external_connections_xyz=[(1, 0)],
    )


def face_centered_cubic3d(a, mass_corner, mass_face, phi=0, theta=0, psi=0):
    return UnitCell3D(
        a1=a * array([1, 0, 0]),
        a2=a * array([sin(phi), cos(phi), 0]),
        a3=a * array([sin(theta)*cos(psi), sin(theta)*sin(psi), cos(theta)]),
        particle_positions=[
            [0, 0, 0],  # corner
            [0, 1/2, 1/2], [1/2, 0, 1/2], [1/2, 1/2, 0],  # faces
        ],
        particle_masses=[
            mass_corner,
            mass_face, mass_face, mass_face
        ],
        internal_connections=[(0, 1), (0, 2), (0, 3)],
        external_connections_x=[(0, 0), (2, 0), (3, 0)],
        external_connections_y=[(0, 0), (3, 0), (1, 0)],
        external_connections_z=[(0, 0), (1, 0), (2, 0)],
        external_connections_xy=[(1, 0), (3, 0)],
        external_connections_yz=[(1, 0), (1, 0)],
        external_connections_xz=[(1, 0), (2, 0)],
        external_connections_xyz=[],
    )


def base_centered_cubic3d(a, mass_corner, mass_face, phi=0, theta=0, psi=0):
    return UnitCell3D(
        a1=a * array([1, 0, 0]),
        a2=a * array([sin(phi), cos(phi), 0]),
        a3=a * array([sin(theta)*cos(psi), sin(theta)*sin(psi), cos(theta)]),
        particle_positions=[
            [0, 0, 0],  # corner
            [1/2, 1/2, 0],  # face
        ],
        particle_masses=[mass_corner, mass_face],
        internal_connections=[(0, 1)],
        external_connections_x=[(0, 0), (1, 0)],
        external_connections_y=[(0, 0), (1, 0)],
        external_connections_z=[(0, 0)],
        external_connections_xy=[(1, 0)],
        external_connections_yz=[],
        external_connections_xz=[],
        external_connections_xyz=[],
    )
