import numpy as np

class CIRModel:
    # Modèle de Cox-Ingersoll-Ross (CIR) pour le taux court :
    # dr = kappa*(theta - r) dt + sigma*sqrt(r) dW
    # Compatible avec PDESolver Crank–Nicolson + Thomas

    def __init__(self, kappa, theta, sigma, lam=0.0, payoff=None):
        # kappa : vitesse de réversion
        # theta : niveau moyen
        # sigma : volatilité
        # lam : lambda pour ajustement du drift
        # payoff : OptionPayoff (vanilla)
        self.kappa = kappa
        self.theta = theta
        self.sigma_param = sigma
        self.lam = lam
        self.payoff_obj = payoff

    # Coefficient de diffusion PDE : aProc = 0.5 * sigma^2 * r
    def a(self, t, r):
        r = np.maximum(r, 0.0)  # éviter racine de négatif
        return 0.5 * (self.sigma_param**2) * r

    # Coefficient de convection PDE : bProc = kappa*(theta - r) - lambda*sigma*sqrt(r)
    def b(self, t, r):
        r = np.maximum(r, 0.0)
        lambda_term = self.lam * np.sqrt(r) / self.sigma_param if self.sigma_param != 0 else 0.0
        return self.kappa * (self.theta - r) - lambda_term * self.sigma_param * np.sqrt(r)

    # Coefficient de discount PDE : cProc = r
    def c(self, t, r):
        return r

    # Condition terminale / payoff
    def payoff(self, r):
        r = np.array(r, dtype=float)
        if self.payoff_obj is not None:
            return self.payoff_obj.payoff(r)
        else:
            # fallback : call sur r
            return np.maximum(r, 0.0)

    # Condition aux bords gauche : r = 0
    def boundary_left(self, tau):
        S_min = 0.0
        if self.payoff_obj is not None:
            return self.payoff_obj.payoff(S_min)
        else:
            return np.maximum(S_min, 0.0)

    # Condition aux bords droite : r = xmax
    def boundary_right(self, tau, S_max):
        if self.payoff_obj is not None:
            return self.payoff_obj.payoff(S_max)
        else:
            return np.maximum(S_max, 0.0)
