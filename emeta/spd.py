# +
import torch
from math import pi


def bordered(m, c, r, d):
    if r is None:
        r = c.T
    mm = torch.cat([torch.cat([m, c], dim=1),
                    torch.cat([r, d], dim=1)])
    return mm


class SPD:

    def __init__(self, matrix=None, epsilon=1e-1):
        self.data = matrix or torch.eye(1)
        self.epsilon = epsilon
        self._cholesky = None
        self._inverse = None

    def cholesky(self):
        if self._cholesky is None:
            self._cholesky = self.data.cholesky()
        return self._cholesky

    def inverse(self):
        if self._inverse is None:
            self._inverse = self.cholesky().cholesky_inverse()
        return self._inverse

    def append_(self, column, diagonal, epsilon=None):
        a = column.view(-1, 1)
        i = torch.as_tensor(diagonal).view(1, 1)
        alpha = self.inverse()@a
        v = (i-a.T@alpha)
        if v < (epsilon if epsilon else self.epsilon):
            return False
        #
        data = bordered(self.data, a, a.T, i)
        try:
            chol = data.cholesky()
            inv = chol.cholesky_inverse()
        except:
            return False
        if data.isnan().any() or chol.isnan().any() or inv.isnan().any():
            raise RuntimeError('nan in SPD!')
        #
        self.data = data
        self._cholesky = chol
        self._inverse = inv
        return True

    def log_prob(self, y):
        _y = torch.as_tensor(y).view(-1, 1)
        f = -0.5*(_y.T@self.inverse()@_y + 2*self.cholesky().diag().log().sum() +
                  _y.size(0)*torch.log(torch.tensor(2*pi)))
        return f
