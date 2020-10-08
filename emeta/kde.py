# +
from .variable import Variable
from .spd import SPD
from collections import Counter
from math import pi
import torch


def discrete(val, delta):
    return tuple(val.div(delta).floor().int().view(-1).tolist())


class GaussianKernel:

    def __init__(self, delta):
        self.delta = torch.as_tensor(delta)

    def __call__(self, x, y):
        d = (x[:, None]-y[None]).div(self.delta).norm(dim=-1)
        k = d.pow(2).neg().exp()
        return k


class History(Variable):

    def __init__(self, var, file=None, stop=float('inf')):
        super().__init__(var, file=file)
        self.requires_update.add(self)
        self.var = var
        self.file = file
        self.history = []
        self.write(f'# {var}', 'w')
        self.stop = stop

    def write(self, msg, mode='a'):
        if self.file:
            with open(self.file, mode) as f:
                f.write(f'{msg}\n')

    def evaluate(self, contex):
        if self.history == []:
            self.update()
        return torch.cat(self.history)

    def update(self):
        if len(self.history) < self.stop:
            t = self.var().clone().detach()
            self.history.append(t)
            self.write(t.tolist())


class Histogram(Variable):

    def __init__(self, var, delta):
        super().__init__(var, delta)
        self.requires_update.add(self)
        self.var = var
        self.delta = torch.as_tensor(delta)
        self.hst = Counter()
        self.fixed = False

    def evaluate(self, context):
        return self.hst[discrete(self.var(contex), self.delta)]

    def update(self):
        if not self.fixed:
            self.hst[discrete(self.var(), self.delta)] += 1.

    def full(self, density=True):
        x = torch.tensor(list(self.hst.keys()))*self.delta
        y = torch.tensor(list(self.hst.values()))
        if density:
            y /= y.sum()*self.delta.prod()
        return x, y

    def save(self, file):
        with open(file, 'w') as f:
            for key, val in self.hst.items():
                f.write(f'{key} : {val}\n')

    def load(self, file):
        with open(file, 'r') as f:
            for line in f:
                key, val = line.split(':')
                self.hst[eval(key)] += eval(val)


class KDE(Histogram):

    def __init__(self, var, kern):
        super().__init__(var, kern.delta)
        self.kern = kern

    def evaluate(self, context):
        X, y = self.full(density=False)
        if X is None:
            return torch.zeros(1)
        x = self.var(context)
        k = self.kern(x, X)
        p = torch.tensor(pi).sqrt()
        kde = k.mul(y).sum(dim=-1) / p
        return kde


class KDR(Variable):

    def __init__(self, var, kern):
        super().__init__(var, kern)
        self.requires_update.add(self)
        self.var = var
        self.kern = kern
        self.fixed = False
        self.inducing = []

    @property
    def X(self):
        return torch.stack(self.inducing)

    def update(self):
        if not self.fixed:
            x = self.var().clone().detach()
            if len(self.inducing) == 0:
                self.inducing.append(x)
                self.k = SPD(epsilon=1e-1)
                self.y = torch.ones(1, 1)
            else:
                k = self.kern(x, self.X)
                if self.k.append_(k, 1.):
                    self.inducing.append(x)
                    self.y = torch.cat([self.y, torch.zeros(1, 1)])
                    k = torch.cat([k, torch.ones(1, 1)], dim=1)
                self.y += k.t()

    def evaluate(self, context):
        if len(self.inducing) == 0:
            return torch.zeros(1)
        x = self.var(context)
        k = self.kern(x, self.X)
        y = self.k.inverse()@self.y
        kde = (k@y).sum(dim=-1)
        return kde
