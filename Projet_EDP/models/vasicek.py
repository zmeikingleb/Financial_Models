import numpy as np

class VasicekModel:
    # Modèle de Vasicek pour le taux court :
    # dr = a*(b - r) dt + sigma dW

    def __init__(self, a, b, sigma, lam=0.0, payoff=None):
        # a : vitesse de réversion
        # b : niveau moyen
        # sigma : volatilité
        # lam : lambda pour ajuster b'
        # payoff : OptionPayoff (vanilla)
        self.a_param = a
        self.b_param = b
        self.sigma = sigma
        self.lam = lam
        self.bprime = b - lam * sigma / a
        self.payoff_obj = payoff

    # Coefficient de diffusion PDE
    def a(self, t, r):
        return 0.5 * self.sigma**2

    # Coefficient de convection PDE
    def b(self, t, r):
        return self.a_param * (self.bprime - r)

    # Coefficient de discount PDE
    def c(self, t, r):
        return r

    # Condition terminale / payoff
    def payoff(self, r):
        r = np.array(r, dtype=float)
        if self.payoff_obj is not None:
            return self.payoff_obj.payoff(r)
        else:
            return np.maximum(r, 0.0)

    # Conditions aux bords
    def boundary_left(self, tau):
        S_min = -1.0
        if self.payoff_obj is not None:
            return self.payoff_obj.payoff(S_min)
        else:
            return np.maximum(S_min, 0.0)

    def boundary_right(self, tau, S_max):
        S_max = 1.0
        if self.payoff_obj is not None:
            return self.payoff_obj.payoff(S_max)
        else:
            return np.maximum(S_max, 0.0)