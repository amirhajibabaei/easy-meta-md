# +
from .variable import Variable
from .spd import SPD
from collections import Counter
from math import pi
import torch


def discrete(val, delta):
    return tuple(val.div(delta).floor().int().view(-1).tolist())


def conformed(x):
    if x.dim() > 1:
        return x
    else:
        return x.view(1, -1)


def dist(x, y):
    _x = conformed(x)
    _y = conformed(y)
    return _x[:, None]-_y[None]


class GaussianKernel:

    def __init__(self, delta):
        self.delta = torch.as_tensor(delta)

    def __call__(self, x, y):
        d = dist(x, y).div(self.delta).norm(dim=-1)
        k = d.pow(2).neg().div(2).exp()
        return k

    def optimize(self, *args, **kwargs):
        return None


class GaussianARD:

    def __init__(self, dim):
        self.trans = torch.distributions.transforms.LowerCholeskyTransform()
        self._param = torch.zeros(dim, dim)

    @property
    def covariance(self):
        chol = self.trans(self._param)
        return chol @ chol.t()

    @property
    def precision(self):
        return self.trans(self._param).cholesky_inverse()

    def __call__(self, x, y):
        r = dist(x, y)
        d = ((r @ self.precision)*r).sum(dim=-1)
        k = d.neg().div(2).exp()
        return k

    def optimize(self, x, y, steps=100, lr=None, noise=0.):
        self._param.requires_grad_(True)
        lr = lr or 1./y.var().sqrt()
        optimizer = torch.optim.LBFGS([self._param], lr=lr)
        for _ in range(steps):
            def closure():
                optimizer.zero_grad()
                matrix = self(x, x) + noise*torch.eye(x.size(0))
                cholesky = matrix.cholesky()
                inverse = cholesky.cholesky_inverse()
                mu = inverse @ y
                loss = (y.T @ mu + 2*cholesky.diag().log().sum())
                loss.backward()
                return loss
            optimizer.step(closure)
        self._param.requires_grad_(False)
        return self(x, x)


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
        p = torch.tensor(2*pi).sqrt()
        kde = k.mul(y).sum(dim=-1) / p
        return kde


class KDR(Variable):

    def __init__(self, var, kern, epsilon=0.1, normalize=True):
        super().__init__(var, kern)
        self.requires_update.add(self)
        self.var = var
        self.kern = kern
        self.epsilon = epsilon
        self.normalize = normalize
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
                self.k = SPD(epsilon=self.epsilon)
                self.y = torch.ones(1, 1)
            else:
                k = self.kern(x, self.X)
                inv = self.k.inverse()
                if self.k.append_(k, 1.):
                    self.inducing.append(x)
                    y = k@inv@self.y  # y(x) -> self.y
                    self.y = torch.cat([self.y, y.view(1, 1)])
                    k = torch.cat([k, torch.ones(1, 1)], dim=1)
                if self.normalize:
                    k = k / (k@self.k.inverse()@k.t()).sqrt().view([])
                self.y += k.t()

    def optimize(self, **kwargs):
        opt = self.kern.optimize(self.X, self.y, **kwargs)
        if opt is not None:
            self.k.data = opt

    def evaluate(self, context):
        if len(self.inducing) == 0:
            return torch.zeros(1)
        x = self.var(context)
        k = self.kern(x, self.X)
        if self.normalize:
            k = k / (k@self.k.inverse()@k.t()).diag()[:, None].sqrt()
        y = self.k.inverse()@self.y
        kde = (k@y).sum(dim=-1)
        return kde
