from numpy import array
import numpy as np


class BlochVector:
    def __init__(self, q, N):
        """Creates the Bloch Vector q from m = [m0, m1, m2, ...], which
        represents
                m0/N0 * b0 + m1/N1 * b1 + ...
        :param q: an interable of integers
        """
        self.q = array(q, dtype=np.int)
        self.N = array(N, dtype=np.int)

    def __eq__(self, other):
        if not isinstance(other, BlochVector):
            return False
        else:
            return (
                self.canonical_form_mod_N() == other.canonical_form_mod_N() and
                self.N.all() == other.N.all()
            )

    def __hash__(self):
        return int(self.canonical_form_mod_N()[0])

    def __cmp__(self, other):
        return self == other

    def __neg__(self):
        return BlochVector(-self.q, self.N)

    def __add__(self, other):
        return BlochVector(self.q + other.q, self.N)

    def __sub__(self, other):
        return BlochVector(self.q - other.q, self.N)

    def __mul__(self, other):
        return BlochVector(other * self.q, self.N)

    def __getitem__(self, item):
        return self.canonical_form()[item]

    def __len__(self):
        return len(self.q)

    def __abs__(self):
        return np.linalg.norm(self.canonical_form(), ord=2)

    def __str__(self):
        return str(self.canonical_form())

    def __lt__(self, other):
        c_self = self.canonical_form_mod_N()
        c_other = other.canonical_form_mod_N()
        return c_self < c_other

    def canonical_form_mod_N(self):
        q_canon = np.mod(self.q, self.N, dtype=np.int)
        return list(q_canon)

    def canonical_form(self):
        return self.canonical_form_mod_N() / self.N

    def dot(self, other):
        assert len(other) == len(self)
        d = sum(map(lambda a: a[0]*a[1], zip(self.canonical_form(), other)), 0)
        return d
