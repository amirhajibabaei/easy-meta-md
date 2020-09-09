# +
from ase.calculators.calculator import Calculator, all_changes
from emeta.atomic import Data
import torch


class Biased(Calculator):

    def __init__(self, bias, calc, logfile='biased.log'):
        super().__init__()
        self.bias = bias
        self._calc = calc
        self.logfile = logfile
        self.log(f'# bias = {bias}', 'w')
        self.log(f'# energy bias')

    def log(self, mssge, mode='a'):
        if self.logfile:
            with open(self.logfile, mode) as f:
                f.write(f'{mssge}\n')

    @property
    def implemented_properties(self):
        return self._calc.implemented_properties

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self._calc.calculate(self.atoms)
        self.results = self._calc.results
        data = Data(self.atoms)

        # energy & forces
        e = self.bias(data)
        self.log('{} {}'.format(self.results['energy'], float(e)))
        e.backward()
        f = -data.pos.grad.detach().numpy()
        self.results['energy'] += float(e)
        self.results['forces'] += f

        # stress
        if 'stress' in self.results.keys():
            s1 = -(f[:, None]*atoms.positions[..., None]).sum(axis=0)
            if data.cell.grad is not None:
                c = data.cell.grad.detach().numpy()
                s2 = (c[:, None]*atoms.cell[..., None]).sum(axis=0)
            else:
                s2 = 0
            try:
                volume = atoms.get_volume()
            except ValueError:
                volume = -2.  # here stress2=0, thus trace(stress) = virial (?)
            stress = (s1 + s2)/volume
            self.results['stress'] += stress.flat[[0, 4, 8, 5, 2, 1]]


def log_to_fig(file='biased.log'):
    import numpy as np
    import pylab as plt
    e, b = np.loadtxt(file, unpack=True)
    fig, axe = plt.subplots(1, 1)
    #
    color = 'blue'
    axe.plot(e, color=color)
    axe.set_xlabel('step')
    axe.set_ylabel('potential', color=color)
    #
    color = 'red'
    axb = axe.twinx()
    axb.plot(b, color=color)
    axb.set_ylabel('bias', color=color)
    return fig
