# +
from emeta.variable import Variable


class Kernel(Variable):

    def __init__(self, u, v, *args, **kwargs):
        super().__init__(u, v, *args, **kwargs)
        self.u = u
        self.v = v

    def eval(self, *args, **kwargs):
        x1 = self.u(*args, **kwargs)
        x2 = self.v(*args, **kwargs)
        return x1, x2


class Dist(Kernel):

    def __init__(self, u, v, scale):
        super().__init__(u, v, scale)
        self.scale = scale

    def eval(self, *args, **kwargs):
        x1, x2 = super().eval(*args, **kwargs)
        delta = (x1-x2).div(self.scale)
        dim = tuple(range(abs(x1.dim()-x2.dim()), max(x1.dim(), x2.dim())))
        if len(dim) > 0:
            delta = delta.norm(dim=dim)
        else:
            delta = delta.abs()
        return delta


class Gaussian(Dist):

    def __init__(self, u, v, scale):
        super().__init__(u, v, scale)

    def eval(self, *args, **kwargs):
        return super().eval(*args, **kwargs).pow(2).neg().exp()
