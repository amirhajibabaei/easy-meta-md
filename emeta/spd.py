# +
import torch
from math import pi


def bordered(m, c, r, d):
    if r is None:
        r = c.T
    mm = torch.cat([torch.cat([m, c], dim=1),
                    torch.cat([r, d], dim=1)])
    return mm


class SPD(torch.Tensor):

    def __init__(self, data, lbound=1e-3):
        self.data = data
        self.lbound = lbound
        self._inverse = None

    def inverse(self):
        if self._inverse is None:
            self._inverse = self.data.cholesky().cholesky_inverse()
        return self._inverse

    def append_(self, column, diagonal, lbound=None):
        a = column.view(-1, 1)
        i = torch.as_tensor(diagonal).view(1, 1)
        alpha = self.inverse()@a
        v = (i-a.T@alpha)
        if v < (lbound if lbound else self.lbound):
            return False
        beta = v.sqrt()
        self.data = bordered(self.data, a, a.T, i)
        self._inverse = bordered(self.inverse() + (alpha@alpha.T)/v,
                                 -alpha/v, None, torch.ones(1, 1)/v)
        return True

    def pop_(self, i):
        k = torch.cat([torch.arange(i), torch.arange(i+1, self.size(0))])
        self.data = self.data.index_select(0, k).index_select(1, k)
        alpha = self._inverse[i].index_select(0, k).view(-1, 1)
        beta = self._inverse[i, i]
        self._inverse = (self._inverse.index_select(0, k).index_select(1, k) -
                         alpha@alpha.T/beta)

    def test(self):
        a = (self.data@self.inverse() - torch.eye(self.size(0))).abs().max()
        return a


class CholSPD(torch.Tensor):

    def __init__(self, data, lbound=1e-3):
        self.data = data
        self.lbound = lbound
        self._cholesky = None

    def cholesky(self):
        if self._cholesky is None:
            self._cholesky = self.data.cholesky()
            self._inverse_of_cholesky = self._cholesky.inverse()
            self._inverse = self._cholesky.cholesky_inverse()
        return self._cholesky

    def inverse(self):
        if self._cholesky is None:
            self.cholesky()
        return self._inverse

    def append_(self, column, diagonal, lbound=None):
        a = column.view(-1, 1)
        i = torch.as_tensor(diagonal).view(1, 1)
        alpha = self.inverse()@a
        v = (i-a.T@alpha)
        if v < (lbound if lbound else self.lbound):
            return False
        beta = v.sqrt()
        self.data = bordered(self.data, a, a.T, i)
        self._cholesky = bordered(self._cholesky, torch.zeros_like(a),
                                  (self._inverse_of_cholesky@a).T, beta)
        self._inverse_of_cholesky = bordered(self._inverse_of_cholesky,
                                             torch.zeros_like(a),
                                             -alpha.T/beta, 1./beta)
        self._inverse = bordered(self._inverse + (alpha@alpha.T)/v,
                                 -alpha/v, None, torch.ones(1, 1)/v)
        return True

    def log_prob(self, _y):
        y = torch.as_tensor(_y).view(-1, 1)
        f = -0.5*(y.T@self.inverse()@y + 2*self._cholesky.diag().log().sum() +
                  y.size(0)*torch.log(torch.tensor(2*pi)))
        return f
