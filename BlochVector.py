import numpy as np
from PeriodicPosition import PeriodicPosition


class BlochVector(PeriodicPosition):
    def __init__(self, q, N):
        """Creates the Bloch Vector q from m = [m0, m1, m2, ...], which
        represents
                m0/N0 * b0 + m1/N1 * b1 + ...
        :param q: an interable of integers
        """
        super(BlochVector, self).__init__(p=q, N=N)

    def canonical_position(self):
        sup = super(BlochVector, self).canonical_position()
        return np.mod(sup, self.N) / self.N
