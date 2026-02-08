from .vanilla import VanillaPayoff
from .exotic import ExoticPayoff
import numpy as np

class OptionPayoff:
    # Payoff d'option (vanilla ou exotique géré par le modèle PDE)
    def __init__(self, K, option_type, exotic=False,
                 barrier=None, knockout=False, knockout_barrier=None,
                 barrier_type="down"):
        # barrier_type : "down", "up", "double"
        self.K = K
        self.option_type = option_type.lower()
        self.exotic = exotic

        self.barrier = barrier
        self.knockout = knockout
        self.knockout_barrier = knockout_barrier
        self.barrier_type = barrier_type

    def payoff(self, S):
        S = np.array(S, dtype=float)
        if self.option_type == "call":
            return np.maximum(S - self.K, 0.0)
        else:
            return np.maximum(self.K - S, 0.0)