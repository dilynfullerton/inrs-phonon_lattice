from matplotlib import pyplot as plt
from numpy.linalg import norm
import itertools as it
from phonon_lattice import *


# Simple 1D Harmonic oscillator lattice
# ----------------------------------------------------------------
K = 1  # Spring constant mass * omega0**2
NX = 4  # Number of particles along line
NY = 4
NZ = 4


# Construct force matrix
def get_c_matrix_simple_harmonic_interaction(k):
    def c_matrix(lattice, k1, x1, p1, k2, x2, p2):
        if (k1, x1, p1.all()) == (k2, x2, p2.all()):
            return k * lattice.num_connections(k1, p1)
        elif x1 == x2 and lattice.are_connected(k1, p1, k2, p2):
            return -1/2 * k
        else:
            return 0
    return c_matrix


def get_c_matrix_coulomb_interaction(g):
    def c_matrix(lattice, k1, x1, p1, k2, x2, p2):
        if not lattice.are_connected(k1, p1, k2, p2):
            return 0  # Only interact with neighbors

        def sterm(ki, pi):
            if (ki, pi.all()) == (k1, p1.all()):
                return 0
            disp = lattice.displacement(ki, pi, k1, p1)
            t1 = -g * 3 * disp[x1] * disp[x2] / norm(disp, ord=2)**5
            if x1 != x2:
                return t1
            else:
                return t1 + g / norm(disp, ord=2)**3

        if k1 == k2:
            s = 0
            for ki, pi in it.product(range(lattice.M), lattice.unit_cells()):
                s += sterm(ki, pi)
            return s
        else:
            return -sterm(k2, p2)
    return c_matrix


# Construct unit cell, a simple line with two masses
MASS_Ga = 69.723  # [amu]
MASS_As = 74.922  # [amu]
A = 1  # Unit cell width
G = 1  # e^2 / (4 pi epsilon0)

GaAs = UnitCell3D(
    a1=[A, 0, 0], a2=[0, A, 0], a3=[0, 0, A],
    particle_positions=[
        # corners
        [0, 0, 0],
        # faces
        [1/2, 1/2, 0], [1/2, 0, 1/2], [0, 1/2, 1/2],
        # center
        [1/4, 1/4, 1/4], [3/4, 3/4, 1/4], [3/4, 1/4, 3/4], [1/4, 3/4, 3/4]
    ],
    particle_masses=[MASS_Ga]*4+[MASS_As]*4,
    internal_connections=[
        (0, 4), (1, 4), (2, 4), (3, 4),
        (1, 5), (2, 6), (3, 7)
    ],
    external_connections_x=[(5, 3), (6, 3)],
    external_connections_y=[(7, 2), (5, 2)],
    external_connections_z=[(6, 1), (7, 1)],
    external_connections_xy=[(5, 0)],
    external_connections_xz=[(6, 0)],
    external_connections_yz=[(7, 0)],
)

# Construct lattice
lat = PhononLattice3D(
    unit_cell=GaAs, N_x=NX, N_y=NY, N_z=NZ,
    c_matrix=get_c_matrix_coulomb_interaction(g=G)
)

# Plot eigenvalues vs. q
xdats = [[] for i in range(3)]
ydats = [[] for i in range(lat.dim_d)]
labs = [
    'omega_{q, ' + '{}'.format(v) + '}^2' for v in range(lat.dim_d)]
for q in lat._A():
    for i in range(3):
        xdats[i].append(q[i])
    for v in range(lat.dim_d):
        ydats[v].append(lat.omega2(q, v))

for v in range(lat.dim_d):
    ydats[v] = [q - ydats[v][0] for q in ydats[v]]

plots = sorted([[(x, y, lab) for y, lab in zip(ydats, labs)] for x in xdats])

for i in range(3):
    fig, ax = plt.subplots(1, 1)
    for xdat, ydat, lab in plots[i]:
        ax.plot(xdat, ydat, '-', label=lab)
    ax.set_xlabel('Bloch wavevector, q[{}]'.format(i))
    ax.set_ylabel('Eigenfrequency, omega_{q,v}^2')
    plt.show()
