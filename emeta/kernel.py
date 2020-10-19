# +
from math import pi
import torch


class Kernel:

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x, y):
        assert x.dim() == 2 and y.dim() == 2
        assert x.size(1) == self.dim and y.size(1) == self.dim
        return self.evaluate(x, y)

    def evaluate(self, x, y):
        raise NotImplementedError('implement in a subclass')


class Stationary(Kernel):

    def __init__(self, dim, scale=1.):
        super().__init__(dim)
        if hasattr(scale, '__call__'):
            self.scale = scale
        else:
            self.scale = lambda: scale

    def evaluate(self, x, y):
        return (x[:, None]-y[None])/self.scale()

    @property
    def dvol(self):
        scale = torch.as_tensor(self.scale()).view(-1)
        if scale.size(0) == 1:
            return scale.pow(self.dim).view([])
        else:
            return scale.prod().view([])


class Distance(Stationary):

    def __init__(self, dim, scale=1., norm=None):
        super().__init__(dim, scale=scale)
        self.norm = norm

    def evaluate(self, x, y):
        return super().evaluate(x, y).norm(self.norm, dim=2)


class Gaussian(Distance):

    def __init__(self, dim, scale=1., norm=None):
        super().__init__(dim, scale=scale, norm=norm)

    def evaluate(self, x, y):
        return super().evaluate(x, y).pow(2).div(2).neg().exp()

    @property
    def normalization(self):
        return torch.tensor(2*pi).pow(self.dim).sqrt()*self.dvol
