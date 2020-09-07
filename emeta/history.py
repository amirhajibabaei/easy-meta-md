# +
from emeta.variable import Variable
import torch


class History(Variable):

    def __init__(self, var, file='history.txt'):
        self.var = var
        self.args = (var,)
        self.kwargs = dict(file=file)
        self.history = []
        self.write(f'# {var}', 'w')

    def write(self, msg, mode='a'):
        if self.kwargs['file']:
            with open(self.kwargs['file'], mode) as f:
                f.write(f'{msg}\n')

    def eval(self, *args, append=True, **kwargs):
        if append:
            t = self.var(*args, *kwargs).clone().detach()
            self.history.append(t)
            self.write(t.tolist())
        return torch.stack(self.history)
