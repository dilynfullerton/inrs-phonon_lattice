from matplotlib import pyplot as plt
from PhononLattice import *
from UnitCell import *
from PhononLattice import get_c_matrix_coulomb_interaction

# UNIT_MASS_KG = 1
# UNIT_LENGTH_M = 1
# UNIT_CHARGE_C = 1
# UNIT_TIME_S = 1
#
# UNIT_LENGTH_CM = UNIT_LENGTH_M / 100
# UNIT_MASS_G = 1/1000 * UNIT_MASS_KG
# UNIT_MASS_AMU = 1.6605e-27 * UNIT_MASS_KG
# UNIT_CHARGE_E = 1.602e-19 * UNIT_CHARGE_C
# UNIT_CAPACITANCE_F = UNIT_TIME_S**2 * UNIT_CHARGE_C**2 / UNIT_LENGTH_M**2 / UNIT_MASS_KG

UNIT_MASS_AMU = 1/100
UNIT_CHARGE_E = 1
UNIT_DENSITY_G_CM3 = 1
UNIT_PERMITIVITY_EPSILON_0 = 1e12

MASS_Cd = 112.414 * UNIT_MASS_AMU
MASS_S = 32.06 * UNIT_MASS_AMU
DENSITY_CdS = 4.826 * UNIT_DENSITY_G_CM3
# uc_CdS = zincblende3d(
#     a=DENSITY_CdS,
#     a_is_density=True,
#     mass1=MASS_Cd,
#     charge1=+2*UNIT_CHARGE_E,
#     mass2=MASS_S,
#     charge2=-2*UNIT_CHARGE_E,
#     phi=0, psi=0, theta=0,
# )
uc_CdS = zincblende3d(
    a=1,
    mass1=1,
    mass2=10,
    charge1=1,
    charge2=-1,
)

NX = 40  # Number of particles along line
NY = 2
NZ = 2

EPSILON_0 = 8.854e-12 * UNIT_PERMITIVITY_EPSILON_0
K_COULOMB = 1 / 4 / np.pi / EPSILON_0
lat_CdS = PhononLattice3D(
    unit_cell=uc_CdS,
    N_x=NX, N_y=NY, N_z=NZ,
    # c_matrix=get_c_matrix_coulomb_interaction(K_COULOMB),
    c_matrix=get_c_matrix_coulomb_interaction(1),
)


# Plot eigenvalues vs. q
lat = lat_CdS
xdat = []
ydats = [[] for v in range(lat.num_modes)]
labs = [
    '$\omega_{q, ' + '{}'.format(v) + '}^2$' for v in range(lat.num_modes)]
for q in sorted(lat.operator_q_vectors()):
    qdisp = q.position()
    if qdisp[0] == 0:
        continue
    if qdisp[1] != 0 or qdisp[2] != 0:
        continue
    print('qx={:4.2}'.format(float(qdisp[0])))
    xdat.append(qdisp[0])
    for v in range(lat.num_modes):
        omega2 = lat.omega2(q, v)
        print('  omega2={:8.4}'.format(omega2))
        ydats[v].append(omega2)

plots0 = [(xdat, ydat, lab) for ydat, lab in zip(ydats, labs)]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig, ax0 = plt.subplots(1, 1)
for xdat, ydat, lab in plots0:
    ax0.plot(xdat, ydat, '-', label=lab)

ax0.set_xlabel('Bloch wavevector along $x$, $q_x$')
ax0.set_ylabel('Eigenfrequency, $\omega_{q,v}^2$')
plt.show()
