# +
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
    for _ in range(steps):
        try:
            energy().backward()
        except:
            pass
        for x in energy.params:
            x._force = x.force
            x.add((x.dot + dt*x.force/2)*dt)
        energy().backward()
        traj.write(f'{energy().data} ')
        for x in energy.params:
            x.dot_add(dt*(x._force+x.force)/2)
            traj.write(f'{x().data} {x.dot} ')
        traj.write('\n')
    traj.close()
