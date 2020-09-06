# +
from emeta.variable import Variable
import torch


class History(Variable):

    def __init__(self, var):
        self.var = var
        self.args = (var,)
        self.history = []

    def eval(self, *args, append=True, **kwargs):
        if append:
            self.history.append(self.var(*args, *kwargs).clone().detach())
        return torch.stack(self.history)
