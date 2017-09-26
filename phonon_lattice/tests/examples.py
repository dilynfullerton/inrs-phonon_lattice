from phonon_lattice import *
from matplotlib import pyplot as plt


# Simple 1D Harmonic oscillator lattice
# ----------------------------------------------------------------
K = 1  # Spring constant mass * omega0**2
NX = 10  # Number of particles along line


# Construct force matrix
def get_c_matrix_simple_harmonic_interaction(k):
    def c_matrix(lattice, k1, x1, p1, k2, x2, p2):
        if (k1, x1, p1) == (k2, x2, p2):
            return k * lattice.num_connections(k1, p1)
        elif x1 == x2 and lattice.are_connected(k1, p1, k2, p2):
            return -1/2 * k
        else:
            return 0
    return c_matrix


# Construct unit cell, a simple line
uc = UnitCell1D(
    a1=[1], particle_positions=[0], particle_masses=[1],
    internal_connections=[], external_connections_x=[(0, 0)]
)

# Construct lattice
lat = PhononLattice1D(
    unit_cell=uc, N_x=NX,
    c_matrix=get_c_matrix_simple_harmonic_interaction(k=K)
)

# Plot eigenvalues vs. q
xdat = []
ydat = []
for q in lat.operator_q_vectors():
    xdat.append(q)
    ydat.append(lat.omega2(q, 0))

fig, ax = plt.subplots(1, 1)
ax.plot(xdat, ydat, '-')
ax.set_xlabel('Bloch wavevector, q')
ax.set_ylabel('Eigenfrequency, omega_{q,1}^2')
plt.show()
