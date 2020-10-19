# +
from emeta.variable import Variable
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
        return torch.stack(self.data)

    @property
    def weights(self):
        return None

    def evaluate(self, context):
        input = torch.as_tensor(self.var(context))
        inducing = self.inducing
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
