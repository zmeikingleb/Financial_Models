import numpy as np

class VanillaPayoff:
    # Payoff Vanilla pour option call ou put européen
    def __init__(self, K, option_type="call"):
        self.K = K
        self.option_type = option_type.lower()

    def payoff(self, S, S_path=None, S_others=None):
        # Arguments S_path et S_others ajoutés pour compatibilité avec OptionPayoff
        # Calcul du payoff
        if self.option_type == "call":
            return np.maximum(S - self.K, 0.0)
        else:
            return np.maximum(self.K - S, 0.0)