from PeriodicPosition import PeriodicPosition


class BlochVector(PeriodicPosition):
    def __init__(self, p, N=None):
        """Creates the Bloch Vector q from m = [m0, m1, m2, ...], which
        represents
                m0/N0 * b0 + m1/N1 * b1 + ...
        :param p: an interable of integers
        """
        super(BlochVector, self).__init__(p=p, N=N)

    def position(self):
        return self.bloch_position()

    def displacement(self):
        return self.bloch_displacement()

    def bloch_position(self):
        return super(BlochVector, self).position() / self.N

    def bloch_displacement(self):
        return super(BlochVector, self).displacement() / self.N
