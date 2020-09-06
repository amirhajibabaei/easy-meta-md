# +
from emeta.variable import Variable


class Kernel(Variable):
    pass

    def __call__(self, x1, x2):
        dim = tuple(range(abs(x1.dim()-x2.dim()), max(x1.dim(), x2.dim())))
        return self.eval(x1, x2, dim=dim)


class Gaussian(Kernel):

    def __init__(self, scale):
        self.scale = scale

    def eval(self, x1, x2, dim):
        return (x1-x2).div(self.scale).pow(2).sum(dim=dim).neg().exp()
