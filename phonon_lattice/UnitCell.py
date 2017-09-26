import numpy as np
from numpy import array, dot


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
            internal_connections,
            external_connections,
    ):
        self.a_vectors = [array(a) for a in a_vectors]
        self.a_matrix = np.hstack(self.a_vectors)
        self.dim = len(a_vectors)
        assert self.a_matrix.shape == (self.dim, self.dim)
        self.particle_positions = [array(pos) for pos in particle_positions]
        self.particle_masses = particle_masses
        self.internal_connections = _all_connections(internal_connections)
        self.external_connections = external_connections
        self.num_particles = len(self.particle_positions)

    def num_connections(self, k):
        nc = 0
        for ic in self.internal_connections:
            if ic[0] == k:
                nc += 1
        for ec_arr in self.external_connections:
            for ec in ec_arr:
                if ec[0] == k:
                    nc += 1
        return nc

    def connected_int(self, k1, k2):
        return (k1, k2) in self.internal_connections

    def connected_ext(self, k1, k2, axis):
        return (k1, k2) in self.external_connections[axis]

    def mass(self, k):
        return self.particle_masses[k]

    def position(self, k, p):
        return dot(self.a_matrix, self.particle_positions[k] + p)


class UnitCell1D(UnitCell):
    def __init__(
            self, a1,
            particle_positions,
            particle_masses,
            internal_connections,
            external_connections_x,
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
            internal_connections=internal_connections,
            external_connections=[
                external_connections_x,
            ]
        )
        self.a1 = self.a_vectors[0]
        self.a2 = self.a_vectors[1]

    def connected_x(self, k1, k2):
        return (k1, k2) in self.external_connections[0]


class UnitCell2D(UnitCell):
    def __init__(
            self, a1, a2,
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
        super(UnitCell2D, self).__init__(
            a_vectors=[a1, a2],
            particle_positions=particle_positions,
            particle_masses=particle_masses,
            internal_connections=internal_connections,
            external_connections=[
                external_connections_x,
                external_connections_y
            ]
        )
        self.a1 = self.a_vectors[0]
        self.a2 = self.a_vectors[1]

    def connected_x(self, k1, k2):
        return (k1, k2) in self.external_connections[0]

    def connected_y(self, k1, k2):
        return (k1, k2) in self.external_connections[1]


class UnitCell3D(UnitCell):
    def __init__(
            self, a1, a2, a3,
            particle_positions,
            particle_masses,
            internal_connections,
            external_connections_x,
            external_connections_y,
            external_connections_z,
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
            internal_connections=internal_connections,
            external_connections=[
                external_connections_x,
                external_connections_y,
                external_connections_z
            ]
        )
        self.a1 = self.a_vectors[0]
        self.a2 = self.a_vectors[1]
        self.a3 = self.a_vectors[2]

    def connected_x(self, k1, k2):
        return (k1, k2) in self.external_connections[0]

    def connected_y(self, k1, k2):
        return (k1, k2) in self.external_connections[1]

    def connected_z(self, k1, k2):
        return (k1, k2) in self.external_connections[2]


