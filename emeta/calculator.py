# +
from ase.calculators.calculator import Calculator, all_changes
import torch


class Biased(Calculator):

    def __init__(self, bias, calc, logfile='biased.log'):
        super().__init__()
        self.bias = bias
        self._calc = calc
        self.logfile = logfile
        self.log('# energy bias', 'w')

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

        atoms = self.atoms
        atoms.xyz = torch.from_numpy(atoms.positions)
        atoms.xyz.requires_grad = True
        atoms.lll = torch.from_numpy(atoms.cell.array)
        atoms.lll.requires_grad = True

        # energy & forces
        e = self.bias(atoms)
        self.log('{} {}'.format(self.results['energy'], float(e)))
        e.backward()
        f = -atoms.xyz.grad.detach().numpy()
        self.results['energy'] += float(e)
        self.results['forces'] += f

        # stress
        if 'stress' in self.results.keys():
            s1 = -(f[:, None]*atoms.positions[..., None]).sum(axis=0)
            if atoms.lll.grad is not None:
                c = atoms.lll.grad.detach().numpy()
                s2 = (c[:, None]*atoms.cell[..., None]).sum(axis=0)
            else:
                s2 = 0
            try:
                volume = atoms.get_volume()
            except ValueError:
                volume = -2.  # here stress2=0, thus trace(stress) = virial (?)
            stress = (s1 + s2)/volume
            self.results['stress'] += stress.flat[[0, 4, 8, 5, 2, 1]]
