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
        self.N = N

    def __eq__(self, other):
        if not isinstance(other, BlochVector):
            return False
        else:
            return (
                self._canonical_int_list() == other._canonical_int_list() and
                self.N == other.N
            )

    def __hash__(self):
        return 0

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
        return self._canonical_int_list() < other._canonical_int_list()

    def _canonical_int_list(self):
        return [qi % Ni for qi, Ni in zip(self.q, self.N)]

    def canonical_form(self):
        return array(
            [qi/Ni for qi, Ni in zip(self._canonical_int_list(), self.N)]
        )

    def dot(self, other):
        assert len(other) == len(self)
        d = sum(map(lambda a: a[0]*a[1], zip(self.canonical_form(), other)), 0)
        return d
