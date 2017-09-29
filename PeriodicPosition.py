import numpy as np
import itertools as it


def periodic_displacement_mod_a(p1, p2, N):
    """Returns the unique vector dp, such that
            p2_i + dp_i = p1_i (mod N_i), and
            -1/2 < dp_i/N_i <= 1/2
    """
    disp_abs = p1 - p2
    disp_rel = np.mod(disp_abs, N)
    for di, i in zip(disp_rel, it.count()):
        if di > N[i] / 2:
            disp_rel[i] = di - N[i]
    return disp_rel


class PeriodicPosition:
    def __init__(self, p, N):
        """Creates the discrete periodic position
                p = p1 * a1 + p2 * a2 + ...
        represented by a list
                [p1, p2, ...]
        This has the property that for two positions p1 and p2,
                p1 == p2  <==>  p1 mod N == p2 mod N,
        namely each component p[j] of p is periodic with period N[j]
        """
        self.N = np.array(N, dtype=np.int)
        self.p = np.array(p)
        self.p0 = np.zeros_like(self.N)

    def __eq__(self, other):
        if not isinstance(other, PeriodicPosition):
            other = PeriodicPosition(other, self.N)
        return self._canonical_form_eq() == other._canonical_form_eq()

    def __hash__(self):
        return hash(self._canonical_form_eq()[0])

    def __cmp__(self, other):
        return self == other

    def __neg__(self):
        return PeriodicPosition(-self.p, self.N)

    def __add__(self, other):
        assert len(self) == len(other)
        if isinstance(other, PeriodicPosition):
            return PeriodicPosition(self.p + other.p, self.N)
        else:
            return PeriodicPosition(self.p + other, self.N)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -(self - other)

    def __mul__(self, other):
        return PeriodicPosition(other * self.p, self.N)

    def __rmul__(self, other):
        return self * other

    def __len__(self):
        return len(self.p)

    def __abs__(self):
        return np.linalg.norm(self.canonical_displacement(), ord=2)

    def __str__(self):
        return str(self.canonical_position())

    def __lt__(self, other):
        if not isinstance(other, PeriodicPosition):
            return False
        else:
            c_self = self._canonical_form_eq()
            c_other = other._canonical_form_eq()
            return c_self < c_other

    def _canonical_form_eq(self):
        """Canonical form for equality comparison
        """
        return list(np.mod(self.p, self.N))

    def canonical_position(self):
        """Returns the smallest positive representation of p, i.e. that in
        which all p_i satisfy 0 <= p_i < N_i
        """
        return np.mod(self.p, self.N)

    def canonical_displacement(self):
        return periodic_displacement_mod_a(p1=self.p, p2=self.p0, N=self.N)

    def dot(self, other):
        assert len(other) == len(self)
        d = sum(map(lambda a: a[0]*a[1], zip(self.canonical_position(), other)),
                0)
        return d
