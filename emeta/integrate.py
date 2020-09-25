# +
import torch


class Traj:

    def __init__(self, file=None, mode='w'):
        if file:
            self.file = open(file, mode)
        else:
            self.file = None

    def write(self, *args, **kwargs):
        if self.file:
            self.file.write(*args, **kwargs)

    def close(self):
        if self.file:
            self.file.close()


def verlet(energy, dt, steps, file=None, mode='w'):
    traj = Traj(file, mode)
    traj.write('# energy ')
    for x in energy.params:
        traj.write(f'{x.name} {x.name}_dot ')
    traj.write('\n')
    try:
        energy().backward()
    except:
        pass
    for _ in range(steps):
        for x in energy.params:
            x._force = x.force
            x.add((x.dot + dt*x.force/2)*dt)
        energy().backward()
        traj.write(f'{energy().data} ')
        for x in energy.params:
            x._dot(dt*(x._force+x.force)/2)
            traj.write(f'{x().data} {x.dot} ')
        traj.write('\n')
        energy.update_history()
    traj.close()


def langevin(energy, dt, f, kt, steps, file=None, mode='w'):
    # J. Chem. Phys. 138, 174102 (2013)
    traj = Traj(file, mode)
    traj.write('# energy ')
    for x in energy.params:
        traj.write(f'{x.name} {x.name}_dot ')
    traj.write('\n')
    alpha = torch.tensor(-f*dt).exp()
    beta = (1-alpha**2).sqrt()*torch.tensor(kt).sqrt()
    try:
        energy().backward()
    except:
        pass
    for _ in range(steps):
        for x in energy.params:
            x._dot(dt*x.force/2)
            x.add(dt*x.dot/2)
        energy().backward()
        for x in energy.params:
            x.dot_(x.dot*alpha + beta*torch.randn(x.dot.shape))
            x.add(dt*x.dot/2)
        energy().backward()
        traj.write(f'{energy().data} ')
        for x in energy.params:
            x._dot(dt*x.force/2)
            traj.write(f'{x().data} {x.dot} ')
        traj.write('\n')
        energy.update_history()
    traj.close()
