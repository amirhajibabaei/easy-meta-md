# +
from emeta.variable import Variable
import torch


class Data:

    def __init__(self, atoms):
        self.pos = torch.from_numpy(atoms.positions)
        self.pos.requires_grad = True
        self.cell = torch.from_numpy(atoms.cell.array)
        self.cell.requires_grad = True
        self.rcell = self.cell.inverse()
        self.pbc = atoms.pbc


class Atomic(Variable):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Cell(Atomic):

    def __init__(self, *args):
        super().__init__(*args)
        self.args = args

    def eval(self, data):
        return data.cell[self.args]


class Pos(Atomic):

    def __init__(self, *args):
        super().__init__(*args)
        self.args = args

    def eval(self, data):
        return data.pos[self.args]


class Scaled(Atomic):

    def __init__(self, vectors):
        super().__init__(vectors)
        self.args = (vectors,)

    def eval(self, data):
        vectors = self.args[0](data)
        return vectors @ data.rcell


class Wrapped(Atomic):

    def __init__(self, vectors):
        super().__init__(vectors)
        self.args = (vectors,)

    def eval(self, data):
        vectors = self.args[0](data)
        scaled = (vectors @ data.rcell) % 1
        return scaled @ data.cell


class Mic(Atomic):

    def __init__(self, vectors):
        super().__init__(vectors)
        self.args = (vectors,)

    def eval(self, data):
        vectors = self.args[0](data)
        scaled = (vectors @ data.rcell) % 1
        mic = torch.where(scaled <= 0.5, scaled, scaled-1.)
        rescaled = mic @ data.cell
        return rescaled
