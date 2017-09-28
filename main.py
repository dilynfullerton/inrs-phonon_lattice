from matplotlib import pyplot as plt
from PhononLattice import *
from UnitCell import *
from PhononLattice import get_c_matrix_coulomb_interaction

# Simple 1D Harmonic oscillator lattice
# ----------------------------------------------------------------

K = 1  # Spring constant mass * omega0**2
M1 = 1
M2 = 10
NX = 40  # Number of particles along line
NY = 40
NZ = 2


# Construct unit cell, a simple line with two masses
fcc = face_centered_cubic3d(
    a=1, mass_corner=M1, mass_face=M2, phi=0, psi=0, theta=0)
bcc = body_centered_cubic3d(
    a=1, mass_corner=M1, mass_center=M2, phi=0, psi=0, theta=0)
sc = simple_cubic3d(
    a=1, mass=M1, phi=0, psi=0, theta=0)
zb = zincblende3d(
    a=1, mass1=M1, mass2=M2, phi=0, psi=0, theta=0)

# # Verify number of connections
# print('num_connections(k={})= {}'.format(0, box2d.num_connections(0)))
# print('num_connections(k={})= {}'.format(1, box2d.num_connections(1)))

# Construct lattice
lat_fcc = PhononLattice3D(
    unit_cell=fcc, N_x=NX, N_y=NY, N_z=NZ,
    c_matrix=get_c_matrix_coulomb_interaction(g=-K),
)
lat_bcc = PhononLattice3D(
    unit_cell=bcc, N_x=NX, N_y=NY, N_z=NZ,
    c_matrix=get_c_matrix_coulomb_interaction(g=-K),
)
lat_sc = PhononLattice3D(
    unit_cell=sc, N_x=NX, N_y=NY, N_z=NZ,
    c_matrix=get_c_matrix_coulomb_interaction(g=-K),
)
lat_zb = PhononLattice3D(
    unit_cell=zb, N_x=NX, N_y=NY, N_z=NZ,
    c_matrix=get_c_matrix_coulomb_interaction(g=-K),
)

# Check D matrix
# dmat = lat._get_matrix_rep_d(q=BlochVector(q=[1, 0], N=[NX, NY]))
# print(dmat)

# Plot eigenvalues vs. q
lattice_plots = []
# for lat in [lat_fcc, lat_bcc, lat_sc, lat_zb]:
for lat in [lat_fcc, lat_zb]:
    xdat = []
    ydats = [[] for v in range(lat.num_modes)]
    labs = [
        'omega_{q, ' + '{}'.format(v) + '}^2' for v in range(lat.num_modes)]
    for q in sorted(lat.operator_q_vectors()):
        if q[0] == 0:
            continue
        if q[1] != q[0] or q[2] != 0:
            continue
        print(q)
        xdat.append(q[0])
        for v in range(lat.num_modes):
            ydats[v].append(lat.omega2(q, v))
    plots0 = [(xdat, ydat, lab) for ydat, lab in zip(ydats, labs)]
    lattice_plots.append(plots0)

# fig, ax = plt.subplots(1, 4)
fig, ax = plt.subplots(1, 2)
for plots0, ax0 in zip(lattice_plots, ax):
    for xdat, ydat, lab in plots0:
        ax0.plot(xdat, ydat, '-', label=lab)
    ax0.set_xlabel('Bloch wavevector, q[0]')
    ax0.set_ylabel('Eigenfrequency, omega_{q,v}^2')
plt.show()
