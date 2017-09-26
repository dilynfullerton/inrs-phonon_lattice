from matplotlib import pyplot as plt
from phonon_lattice import *


# Simple 1D Harmonic oscillator lattice
# ----------------------------------------------------------------
K = 1  # Spring constant mass * omega0**2
NX = 10  # Number of particles along line


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


# Construct unit cell, a simple line with two masses
uc = UnitCell2D(
    a1=[1, 0], a2=[0, 1], particle_positions=[0], particle_masses=[1],
    internal_connections=[], external_connections_x=[(0, 0)],
    external_connections_y=[],
)

# Construct lattice
lat = PhononLattice2D(
    unit_cell=uc, N_x=NX, N_y=1,
    c_matrix=get_c_matrix_simple_harmonic_interaction(k=K)
)

# Plot eigenvalues vs. q
xdat = []
ydati = [[] for i in range(lat.dim_d)]
labi = [
    'omega_{q, ' + '{}'.format(v) + '}^2' for v in range(lat.dim_d)]
for q in lat.operator_q_vectors():
    print(q.canonical_form())
    xdat.append(abs(q))
    for v in range(lat.dim_d):
        ydati[v].append(lat.omega2(q, v))

plots = [(xdat, ydat, lab) for ydat, lab in zip(ydati, labi)]

fig, ax = plt.subplots(1, 1)
for xdat, ydat, lab in plots:
    ax.plot(xdat, ydat, '-', label=lab)
ax.set_xlabel('Bloch wavevector, q')
ax.set_ylabel('Eigenfrequency, omega_{q,1}^2')
ax.legend()
plt.show()
