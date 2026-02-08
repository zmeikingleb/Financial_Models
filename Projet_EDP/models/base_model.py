import numpy as np
from abc import ABC, abstractmethod

class PDEModel(ABC):
    #Classe abstraite pour tous les modèles d'EDP financiers.

    @abstractmethod
    def a(self, t, x):
        #Coefficient de la dérivée seconde
        pass

    @abstractmethod
    def b(self, t, x):
        #Coefficient de la dérivée première
        pass

    @abstractmethod
    def c(self, t, x):
        #Coefficient du terme -c*V
        pass

    @abstractmethod
    def d(self, t, x):
        #Terme additionnel
        pass

    @abstractmethod
    def payoff(self, x):
        #Condition terminale
        pass

    def boundary_left(self, t):
        #Condition au bord gauche
        return 0.0

    def boundary_right(self, t):
        #Condition au bord droit
        return 0.0