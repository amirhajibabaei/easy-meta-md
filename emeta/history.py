# +
from emeta.variable import Variable
import torch


class History(Variable):

    def __init__(self, var, file=None):
        super().__init__(var, file=file)
        self.var = var
        self.file = file
        self.history = []
        self.write(f'# {var}', 'w')

    def write(self, msg, mode='a'):
        if self.file:
            with open(self.file, mode) as f:
                f.write(f'{msg}\n')

    def eval(self, *args, append=True, **kwargs):
        if append:
            t = self.var(*args, *kwargs).clone().detach()
            self.history.append(t)
            self.write(t.tolist())
        return torch.stack(self.history)
