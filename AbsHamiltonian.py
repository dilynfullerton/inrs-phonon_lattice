import itertools as it
from math import sqrt


class ModelSpacePhonon:
    def a(self, q, v):
        raise NotImplementedError

    def _iter_q(self):
        raise NotImplementedError

    def _iter_v(self):
        raise NotImplementedError


class ModelSpaceElectron:
    def c(self, q, v):
        raise NotImplementedError

    def _iter_n(self):
        raise NotImplementedError

    def _iter_k(self):
        raise NotImplementedError


class AbsElectronHamiltonian(ModelSpaceElectron):
    def H_e(self):
        he = 0
        for n, k in it.product(self._iter_n(), self._iter_k()):
            c = self.c(n, k)
            he += self.epsilon(n, k) * c.dag() * c
        return he

    def epsilon(self, n, k):
        raise NotImplementedError


class AbsPhononHamiltonian(ModelSpacePhonon):
    def H_p(self):
        hp = 0
        for q, v in it.product(self._iter_q(), self._iter_v()):
            a = self.a(q, v)
            hp += self.omega(q, v) * (a.dag() * a + 1/2)
        return hp

    def omega(self, q, v):
        raise NotImplementedError


class AbsElectronPhononHamiltonian(ModelSpaceElectron, ModelSpacePhonon):
    def H_ep(self):
        return self._H_ep0() + self._H_ep1()

    def g(self, m, n, v):
        raise NotImplementedError

    def g_DW(self, m, n, v1, v2):
        raise NotImplementedError

    def N_p(self):
        raise NotImplementedError

    def _H_ep0(self):
        hep0 = 0
        for m, n, v in it.product(
                self._iter_n(), self._iter_n(), self._iter_v()):
            g = self.g(m, n, v)
            for k, q in it.product(self._iter_k(), self._iter_q()):
                hep0 += (
                    g(k, q) * self.c(m, k+q).dag() * self.c(n, k) *
                    (self.a(q, v) + self.a(-q, v).dag())
                )
        hep0 /= sqrt(self.N_p())
        return hep0

    def _H_ep1(self):
        hep1 = 0
        for m, n, v1, v2 in it.product(it.product(self._iter_n(), repeat=2),
                                       it.product(self._iter_v(), repeat=2)):
            g_DW = self.g_DW(m, n, v1, v2)
            for k, q1, q2 in it.product(
                    self._iter_k(), self._iter_q(), self._iter_q()):
                hep1 += (
                    g_DW(k, q1, q2) * self.c(m, k+q1+q2).dag() * self.c(n, k) *
                    (self.a(q1, v1) + self.a(-q1, v1).dag()) *
                    (self.a(q2, v2) + self.a(-q2, v2).dag())
                )
        hep1 /= self.N_p()
        return hep1


class AbsHamiltonian(AbsElectronHamiltonian,
                     AbsPhononHamiltonian,
                     AbsElectronPhononHamiltonian):
    """Abstract definition for an electron-phonon lattice, which must
    provide for the creation of the Hamiltonian H, defined by
        H = (1) sum_{n,k}[ epsilon_{n,k} * c_{n,k}^dag * c_{n,k} ] +
            (2) sum_{q,v}[ omega_{q,v} * (a_{q,v}^dag * a_{q,v} + 1/2) ] +
            (3) N_p^(-1/2) * sum_{k,q,m,n,v} [
                    g_{m,n,v}(k,q) * c_{m,k+q}^dag * c_{n,k} *
                    (a_{q,v} + a_{-q,v}^dag)
                ]
            (4) N_p^(-1) * sum_{k,q,q',m,n,v,v'} [
                    g_{m,n,v,v'}^DM(k,q,q') * c_{m,k+q+q'}^dag * c_{n,k} *
                    (a_{q,v} + a_{-q,v}^dag) * (a_{q',v'} + a_{-q',v'}^dag)
                ]
    where the fourth term is optional.
    """
    def H(self):
        return self.H_e() + self.H_p() + self.H_ep()

