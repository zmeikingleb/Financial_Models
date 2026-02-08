import numpy as np
from pde.thomas import thomas
from pde.grid import Grid

class PDESolver:

    # Solveur Crank–Nicolson générique pour PDEModel
    def __init__(self, model, xmin, xmax, Nx, tmin=0.0, tmax=1.0, Nt=100, theta=0.5):
        # model : instance de PDEModel
        # xmin, xmax : domaine spatial
        # Nx : nombre de points spatiaux
        # tmin, tmax : temps
        # Nt : nombre de pas de temps
        # theta : paramètre θ du schéma (0.5 = Crank–Nicolson)
        self.model = model
        self.grid = Grid(xmin, xmax, Nx, tmin, tmax, Nt)
        self.theta = theta

    def solve(self):
        # Récupération des paramètres de grille
        Nx, Nt = self.grid.Nx, self.grid.Nt
        dx, dt = self.grid.dx, self.grid.dt
        x, t = self.grid.x, self.grid.t
        T = self.grid.tmax

        # Initialisation de la solution
        V = np.zeros((Nt, Nx))
        V[-1, :] = self.model.payoff(x)  # condition terminale

        # Boucle temps inversée (de T à 0)
        for n in reversed(range(Nt-1)):

            # Vecteurs tridiagonaux
            a_coeff = np.zeros(Nx-1)  # sous-diagonale
            b_coeff = np.zeros(Nx)    # diagonale
            c_coeff = np.zeros(Nx-1)  # sur-diagonale
            d_vec   = np.zeros(Nx)    # second membre

            # Construction des coefficients internes
            for i in range(1, Nx-1):

                # Coefficients de diffusion et convection
                a = self.model.a(t[n+1], x[i])
                b = self.model.b(t[n+1], x[i])
                c = self.model.c(t[n+1], x[i])

                a_diff = a / dx**2

                # Upwind pour le terme de convection
                if b >= 0:
                    b_m = b / dx
                    b_p = 0.0
                else:
                    b_m = 0.0
                    b_p = -b / dx

                # Coefficients tridiagonaux implicites (θ-scheme)
                a_coeff[i-1] = -self.theta * dt * (a_diff + b_m)
                b_coeff[i]   = 1 + self.theta * dt * (2*a_diff + b_m + b_p + c)
                c_coeff[i]   = -self.theta * dt * (a_diff + b_p)

                # Terme droit explicite
                # --- sécurisation overflow ---
                term = (1-self.theta) * dt
                d_vec[i] = (
                    term*(a_diff+b_m)*V[n+1,i-1]
                    + (1 - term*(2*a_diff+b_m+b_p+c))*V[n+1,i]
                    + term*(a_diff+b_p)*V[n+1,i+1]
                )
                # Clip pour éviter overflow
                d_vec[i] = np.clip(d_vec[i], -1e10, 1e10)

            # Temps restant tau = T - t
            tau = T - t[n]

            # Conditions aux bords
            b_coeff[0]  = 1.0
            d_vec[0]    = self.model.boundary_left(tau)

            b_coeff[-1] = 1.0
            d_vec[-1]   = self.model.boundary_right(tau, x[-1])

            # Résolution tridiagonale
            V[n, :] = thomas(a_coeff, b_coeff, c_coeff, d_vec)

            # --- Barrières absorbantes ---
            # Barrière basse
            if hasattr(self.model, "barrier_left"):
                B_left = self.model.barrier_left()
                if B_left is not None:
                    idx = np.searchsorted(x, B_left)
                    if 0 <= idx < Nx:
                        V[n, idx] = self.model.barrier_value(tau)

            # Barrière haute
            if hasattr(self.model, "barrier_right"):
                B_right = self.model.barrier_right()
                if B_right is not None:
                    idx = np.searchsorted(x, B_right)
                    if 0 <= idx < Nx:
                        V[n, idx] = self.model.barrier_value(tau)

        return V, x, t