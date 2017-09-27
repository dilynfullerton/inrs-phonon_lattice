from qutip import *
from PhononLattice import *
from BlochVector import BlochVector
from UnitCell import *
import numpy as np
from matplotlib import pyplot as plt


class ModelSpace:
    def __init__(self, phonon_lattice):
        self.lat = phonon_lattice
        self._N = self.lat.N
        self.Np = self.lat.Np
        self._dim_space = self.lat.dim_space
        self.num_modes = self.lat.num_modes
        self.dim_cavity = 1
        self.dim_phonon = self.Np * self.num_modes
        self.dim_em = 1
        self.dim = self.dim_cavity + self.dim_phonon + self.dim_em

    def q0(self):
        return BlochVector(q=[0]*self._dim_space, N=self._N)

    def qvectors(self):
        return self.lat.operator_q_vectors()

    def vacuum(self):
        return tensor([fock_dm(2)] * self.dim)

    def zero(self):
        return qzero([2] * self.dim)

    def one(self):
        return qeye([2] * self.dim)

    def a(self):
        """Cavity annihilation operator
        """
        return tensor(
            destroy(2),
            qeye([2] * self.dim_phonon),
            qeye([2] * self.dim_em)
        )

    def b(self, q, v):
        """Phonon annihilation operator for crystal momentum q (BlochVector)
        and mode v
        """
        return tensor(
            qeye([2] * self.dim_cavity),
            self.lat.a(q, v),
            qeye([2] * self.dim_em)
        )

    def sig(self):
        """Electric mode annihilation operator
        """
        return tensor(
            qeye([2] * self.dim_cavity),
            qeye([2] * self.dim_phonon),
            sigmam()
        )

    def ham_ph(self):
        return tensor(
            qzero([2] * self.dim_cavity),
            self.lat.H_p(),
            qzero([2] * self.dim_em)
        )


class Hamiltonian:
    def __init__(self, model_space, omega_cavity, omega_em, g,
                 omega_L, Omega_p,
                 kappa, gamma_e, gamma_e_phi, gamma_v, gamma_v_phi):
        self.ms = model_space
        self.VAC = self.ms.vacuum()
        self.ZERO = self.ms.zero()
        self.ONE = self.ms.one()
        self._a = self.ms.a()
        self._b = self.ms.b
        self._sig = self.ms.sig()
        self._num_a = self._a.dag() * self._a
        self._num_sig = self._sig.dag() * self._sig
        self._x_sig = self._sig + self._sig.dag()

        self._q0 = self.ms.q0()

        # Frequencies
        self.omega_cavity = omega_cavity
        self.omega_em = omega_em
        self.g = g
        self.Omega_p = Omega_p
        self.omega_L = omega_L

        # Loss terms
        self.kappa = kappa
        self.gamma_e = gamma_e
        self.gamma_e_phi = gamma_e_phi
        self.gamma_v = gamma_v
        self.gamma_v_phi = gamma_v_phi

    def h(self):
        return self.h_sys() + self.h_int() + self.h_drive()

    def h_sys(self):
        h_cavity = self.omega_cavity * self._num_a
        h_phonon = self.ms.ham_ph()
        h_em = self.omega_em * self._num_sig
        return h_cavity + h_phonon + h_em

    def h_int(self):
        hint = 0
        for v in range(self.ms.num_modes):
            hint += self._num_sig * (
                self._b(self._q0, v) + self._b(self._q0, v).dag())
        return 1/sqrt(self.ms.Np) * self.g * hint

    def h_drive(self):
        return self.Omega_p * self._x_sig - self.omega_L * self._num_sig

    def c_ops(self):
        cops = [
            sqrt(self.kappa) * self._a,
            sqrt(self.gamma_e) * self._sig,
            sqrt(self.gamma_e_phi) * self._num_sig,
        ]
        for q, v in it.product(self.ms.qvectors(), range(self.ms.num_modes)):
            cops.extend(
                [
                    sqrt(self.gamma_v) * self._b(q, v),
                    sqrt(self.gamma_v_phi) * self._num_b(q, v)
                ]
            )
        return cops

    def _num_b(self, q, v):
        return self._b(q, v).dag() * self._b(q, v)


if __name__ == '__main__':
    M1 = 1
    M2 = 10
    NX = 10
    # K = 44444
    K = 1
    OMEGA_C = K / 400
    OMEGA_L = OMEGA_C * 7
    OMEGA_E = OMEGA_C * 10
    G = 0
    OMEGA_P = OMEGA_C / 10

    KAPPA = 1
    GAMMA_V = KAPPA
    GAMMA_V_PHI = GAMMA_V
    GAMMA_E = 4 * KAPPA
    GAMMA_E_PHI = GAMMA_E

    OMEGA_MIN = OMEGA_C / 2
    OMEGA_MAX = OMEGA_E
    NPTS = 11

    # Make unit cell
    uc = UnitCell1D(
        a1=[1],
        particle_positions=[[0], [1/2]],
        particle_masses=[M1, M2],
        internal_connections=[(0, 1)],
        external_connections_x=[(1, 0)],
    )

    # Make lattice
    lat = PhononLattice1D(
        unit_cell=uc, N_x=NX,
        c_matrix=get_c_matrix_simple_harmonic_interaction(k=K)
    )

    for q in lat.operator_q_vectors():
        for v in range(lat.num_modes):
            print(lat.omega(q, v))

    # Make model space
    ms = ModelSpace(phonon_lattice=lat)

    # Make Hamiltonian
    ham = Hamiltonian(
        model_space=ms,
        omega_cavity=OMEGA_C,
        omega_em=OMEGA_E,
        g=G,
        omega_L=OMEGA_L, Omega_p=OMEGA_P,
        kappa=KAPPA, gamma_e=GAMMA_E, gamma_e_phi=GAMMA_E_PHI,
        gamma_v=GAMMA_V, gamma_v_phi=GAMMA_V_PHI
    )

    # Plot spectrum
    print('Getting spectrum...')
    xdat = np.linspace(OMEGA_MIN, OMEGA_MAX, NPTS)
    ydat = spectrum(H=ham.h(), wlist=xdat, c_ops=ham.c_ops(),
                    a_op=ms.sig().dag(), b_op=ms.sig())
    print('Plotting...')
    fig, ax = plt.subplots(1, 1)
    ax.plot(xdat, ydat)
    plt.show()
