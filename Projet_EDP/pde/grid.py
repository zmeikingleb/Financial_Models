import numpy as np

class Grid:
    def __init__(self, xmin, xmax, Nx, tmin, tmax, Nt):
        self.xmin = xmin
        self.xmax = xmax
        self.Nx = Nx
        self.tmin = tmin
        self.tmax = tmax
        self.Nt = Nt

        self.dx = (xmax - xmin) / (Nx - 1)
        self.dt = (tmax - tmin) / (Nt - 1)

        self.x = np.linspace(xmin, xmax, Nx)
        self.t = np.linspace(tmin, tmax, Nt)