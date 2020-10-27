# +
from .variable import Variable
from .spd import SPD
from collections import Counter
import torch


def discrete(val, scale):
    return tuple(val.div(scale).floor().int().view(-1).tolist())


class Density(Variable):

    def __init__(self, var, kern, as_hist=False):
        super().__init__(var, kern)
        self.requires_update.add(self)
        self.var = var
        self.kern = kern
        self.as_hist = as_hist
        self.data = []

    @property
    def inducing(self):
        if len(self.data) > 0:
            return torch.stack(self.data)
        else:
            return None

    @property
    def weights(self):
        return None

    def evaluate(self, context):
        input = torch.atleast_2d(torch.as_tensor(self.var(context)))
        inducing = self.inducing
        if inducing is None:
            return torch.tensor(0.)
        kern = self.kern(input, inducing)
        weights = self.weights.type(kern.type())
        if weights is None:
            norm = inducing.size(0)
        else:
            kern = kern@weights
            norm = weights.sum()
        if self.as_hist:
            norm = 1./self.kern.dvol
        return kern.sum(dim=1)/(self.kern.normalization*norm)

    def update(self, value=None):
        x = value or self.var().clone().detach()
        self._update(x)

    def _update(self, x):
        self.data.append(x)


class GridKDE(Density):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = Counter()

    @property
    def inducing(self):
        return torch.tensor(list(self.count.keys())).add(0.5)*self.kern.scale()

    @property
    def weights(self):
        return torch.tensor(list(self.count.values())).view(-1, 1)

    def _update(self, x):
        self.count[discrete(x, self.kern.scale())] += 1.


class KDR(Density):

    def __init__(self, *args, dirac=None, epsilon=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.dirac = dirac or self.kern

    @property
    def weights(self):
        return self._w

    def _update(self, x):
        if len(self.data) == 0:
            self.data.append(x)
            self.k = SPD(epsilon=self.epsilon)
            self._w = torch.ones(1, 1)
        else:
            delta = self.dirac(x.view(1, -1), self.inducing)
            inv = self.k.inverse()
            self._w += inv@delta.t()
            k = self.kern(x.view(1, -1), self.inducing)
            if self.k.append_(k, 1.):
                self.data.append(x)
                self._w = torch.cat([self._w, torch.zeros(1, 1)])
