import numpy as np

class ExoticPayoff:
    # Payoffs exotiques : barrier, knockout, asiatique, rainbow
    def __init__(self, K, option_type="call",
                 barrier=None, knockout=False, knockout_barrier=None):
        # Arguments :
        # K : strike
        # option_type : "call" ou "put"
        # barrier : float ou None
        # knockout : bool
        # knockout_barrier : float ou None
        # asian : bool
        self.K = K
        self.option_type = option_type.lower()
        self.barrier = barrier
        self.knockout = knockout
        self.knockout_barrier = knockout_barrier

        # Validation : un seul type exotique à la fois
        exotic_flags = [ knockout or barrier is not None]
        if sum(exotic_flags) > 1:
            raise ValueError("Un seul type d'option exotique à la fois.")

    def payoff(self, S, S_path=None, S_others=None):
        # S peut être un scalaire ou un array
        S = np.array(S, dtype=float)

        # Knockout
        if self.knockout and self.knockout_barrier is not None:
            if self.option_type == "call":
                S = np.where(S >= self.knockout_barrier, 0.0, S)
            else:
                S = np.where(S <= self.knockout_barrier, 0.0, S)

        # Barrier
        elif self.barrier is not None:
            if self.option_type == "call":
                S = np.where(S < self.barrier, 0.0, S)
            else:
                S = np.where(S > self.barrier, 0.0, S)

        # Payoff Vanilla final
        if self.option_type == "call":
            return np.maximum(S - self.K, 0.0)
        else:
            return np.maximum(self.K - S, 0.0)
