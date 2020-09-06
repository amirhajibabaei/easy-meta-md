# +
from emeta.variable import Variable


class Kernel(Variable):

    def __init__(self, u, v):
        self.u = u
        self.v = v

    @property
    def args(self):
        return (self.u, self.v)

    def __call__(self, *args, **kwargs):
        x1 = self.u(*args, **kwargs)
        x2 = self.v(*args, **kwargs)
        dim = tuple(range(abs(x1.dim()-x2.dim()), max(x1.dim(), x2.dim())))
        return self.eval(x1, x2, dim=dim)


class Gaussian(Kernel):

    def __init__(self, u, v, scale):
        super().__init__(u, v)
        self.scale = scale

    @property
    def args(self):
        return (*super().args, self.scale)

    def eval(self, x1, x2, dim):
        return (x1-x2).div(self.scale).pow(2).sum(dim=dim).neg().exp()
