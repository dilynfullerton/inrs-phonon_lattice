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
            return -k
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
line1d = UnitCell1D(
    a1=[1],
    particle_positions=[0, 1/2],
    particle_masses=[.1, 1.],
    internal_connections=[(0, 1)],
    external_connections_x=[(1, 0)]
)

# Verify number of connections
print('num_connections(k={})= {}'.format(0, line1d.num_connections(0)))
print('num_connections(k={})= {}'.format(1, line1d.num_connections(1)))

# Construct lattice
lat = PhononLattice1D(
    unit_cell=line1d, N_x=NX,
    c_matrix=get_c_matrix_simple_harmonic_interaction(k=K)
)

# Plot eigenvalues vs. q
xdat = []
ydats = [[] for v in range(lat.dim_d)]
labs = [
    'omega_{q, ' + '{}'.format(v) + '}^2' for v in range(lat.dim_d)]
for q in lat.q_vectors():
    xdat.append(q[0])
    for v in range(lat.dim_d):
        ydats[v].append(lat.omega2(q, v))

plots = sorted([(xdat, ydat, lab) for ydat, lab in zip(ydats, labs)])

fig, ax = plt.subplots(1, 1)
for xdat, ydat, lab in plots:
    ax.plot(xdat, ydat, '-', label=lab)
ax.set_xlabel('Bloch wavevector, q[0]')
ax.set_ylabel('Eigenfrequency, omega_{q,v}^2')
plt.show()
