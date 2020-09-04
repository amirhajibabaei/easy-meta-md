# +
from emeta.variable import Variable
import torch


class C(Variable):
    """cell"""

    def __init__(self, *args):
        self.args = args

    def eval(self, atoms):
        return atoms.lll[self.args]


class P(Variable):
    """positions"""

    def __init__(self, *args):
        self.args = args

    def eval(self, atoms):
        return atoms.xyz[self.args]


class SP(Variable):
    """scaled positions"""

    def __init__(self, *args):
        self.args = args

    def eval(self, atoms):
        p = atoms.xyz[self.args]
        rc = atoms.cell.reciprocal().array
        r = p@torch.from_numpy(rc)
        r = r % 1.
        return r


class D(Variable):
    """distance"""

    def __init__(self, i, j, mic=False, vector=False):
        self.i = i
        self.j = j
        self.mic = mic
        self.vector = vector
        self.args = (i, j)
        self.kwargs = dict(mic=mic, vector=vector)

    def eval(self, atoms):
        raw = atoms[self.j].position - atoms[self.i].position
        real = atoms.get_distance(self.i, self.j, mic=self.mic, vector=True)
        result = (atoms.xyz[self.j] - atoms.xyz[self.i] +
                  torch.from_numpy(real-raw))
        if not self.vector:
            result = result.norm()
        return result
