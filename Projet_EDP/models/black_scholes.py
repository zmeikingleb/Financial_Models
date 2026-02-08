import numpy as np
from models.base_model import PDEModel
from payoffs.option_payoff import OptionPayoff

class BlackScholesModel:
    # Modèle Black–Scholes pour PDE
    def __init__(self, K, sigma, r, q, payoff):
        self.K = K
        self.sigma = sigma
        self.r = r
        self.q = q
        self.payoff_obj = payoff

        self.barrier_type = payoff.barrier_type
        self.barrier_low = None
        self.barrier_high = None

        if payoff.exotic:
            if self.barrier_type == "down":
                self.barrier_low = payoff.barrier or payoff.knockout_barrier
            elif self.barrier_type == "up":
                self.barrier_high = payoff.barrier or payoff.knockout_barrier
            elif self.barrier_type == "double":
                self.barrier_low, self.barrier_high = payoff.barrier

    # Coefficients PDE
    def a(self, t, x):
        return 0.5 * self.sigma**2 * x**2

    def b(self, t, x):
        return (self.r - self.q) * x

    def c(self, t, x):
        return -self.r

    def payoff(self, x):
        return self.payoff_obj.payoff(x)

    # Conditions aux bords
    def boundary_left(self, tau):
        if self.barrier_low is not None:
            return 0.0
        if self.payoff_obj.option_type == "call":
            return 0.0
        return self.K * np.exp(-self.r * tau)

    def boundary_right(self, tau, x_max):
        if self.barrier_high is not None:
            return 0.0
        if self.payoff_obj.option_type == "call":
            return x_max * np.exp(-self.q * tau) - self.K * np.exp(-self.r * tau)
        return 0.0

    # Interface barrière
    def has_barrier(self):
        return self.barrier_low is not None or self.barrier_high is not None

    def barrier_left(self):
        return self.barrier_low

    def barrier_right(self):
        return self.barrier_high

    def barrier_value(self, tau):
        return 0.0