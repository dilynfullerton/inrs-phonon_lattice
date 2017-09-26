from numpy import array, dot
import numpy as np


class BlochVector:
    def __init__(self, q):
        self.q = array(q)

    def __eq__(self, other):
        if not isinstance(other, BlochVector):
            return False
        else:
            return self.canonical_form().all() == other.canonical_form().all()

    def __hash__(self):
        return int(abs(self))

    def __neg__(self):
        return BlochVector(-self.q)

    def __add__(self, other):
        return BlochVector(self.q + other.q)

    def __sub__(self, other):
        return BlochVector(self.q - other.q)

    def __mul__(self, other):
        return BlochVector(other * self.q)

    def __getitem__(self, item):
        return self.q[item]

    def __len__(self):
        return len(self.q)

    def __abs__(self):
        return np.linalg.norm(self.canonical_form(), ord=2)

    def canonical_form(self):
        return array([qi - int(qi) for qi in self.q])

    def dot(self, other):
        assert len(other) == len(self)
        other = array(other)
        return np.dot(self.q, other)
