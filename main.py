from matplotlib import pyplot as plt
from PhononLattice import *
from UnitCell import *
from PhononLattice import get_c_matrix_coulomb_interaction

UNIT_LENGTH_M = 1
UNIT_MASS_KG = 1
UNIT_CURRENT_A = 1
UNIT_TIME_S = 1

UNIT_LENGTH_CM = UNIT_LENGTH_M / 100
UNIT_MASS_G = 1/1000 * UNIT_MASS_KG
UNIT_MASS_AMU = 1.6605e-27 * UNIT_MASS_KG
UNIT_CHARGE_C = UNIT_TIME_S * UNIT_CURRENT_A
UNIT_CHARGE_E = 1.602e-19 * UNIT_CHARGE_C
UNIT_CAPACITANCE_F = UNIT_TIME_S**4 * UNIT_CURRENT_A**2 / UNIT_LENGTH_M**2 / UNIT_MASS_KG

MASS_Cd = 112.414 * UNIT_MASS_AMU
MASS_S = 32.06 * UNIT_MASS_AMU
DENSITY_CdS = 4.826 * UNIT_MASS_G / UNIT_LENGTH_CM**3
uc_CdS = zincblende3d(
    a=DENSITY_CdS,
    a_is_density=True,
    mass1=MASS_Cd,
    mass2=MASS_S,
    phi=0, psi=0, theta=0,
    particle_charges=[+2 * UNIT_CHARGE_E, -2 * UNIT_CHARGE_E]
)

NX = 40  # Number of particles along line
NY = 2
NZ = 2

EPSILON_0 = 8.854e-12 * UNIT_CAPACITANCE_F / UNIT_LENGTH_M
K_COULOMB = 1 / 4 / np.pi / EPSILON_0
lat_CdS = PhononLattice3D(
    unit_cell=uc_CdS,
    N_x=NX, N_y=NY, N_z=NZ,
    c_matrix=get_c_matrix_coulomb_interaction2(K_COULOMB)
)


# Plot eigenvalues vs. q
lat = lat_CdS
xdat = []
ydats = [[] for v in range(lat.num_modes)]
labs = [
    '$\omega_{q, ' + '{}'.format(v) + '}^2$' for v in range(lat.num_modes)]
for q in sorted(lat.operator_q_vectors()):
    if q[0] == 0:
        continue
    if q[1] != 0 or q[2] != 0:
        continue
    print(q)
    xdat.append(q[0])
    for v in range(lat.num_modes):
        ydats[v].append(lat.omega2(q, v))

plots0 = [(xdat, ydat, lab) for ydat, lab in zip(ydats, labs)]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax0 = plt.subplots(1, 1)
for xdat, ydat, lab in plots0:
    ax0.plot(xdat, ydat, '-', label=lab)

ax0.set_xlabel('Bloch wavevector along $x$, $q_x$')
ax0.set_ylabel('Eigenfrequency, $\omega_{q,v}^2$')
plt.show()
