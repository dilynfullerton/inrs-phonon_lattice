from itertools import product


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
        self.a1 = a1
        self.a2 = a2
        self.particle_positions = particle_positions
        self.particle_masses = particle_masses
        self.internal_connections = internal_connections
        self.external_connections_x = external_connections_x
        self.external_connections_y = external_connections_y


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


class Lattice2D:
    def __init__(self, unit_cell, N_x, N_y, c_matrix):
        self.Nx = N_x
        self.Ny = N_y
