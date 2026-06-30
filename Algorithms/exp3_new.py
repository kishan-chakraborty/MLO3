"""
This is a variation of EXP3 with decaying exploration rate with the intention to do experimentation
for congestion game. I want the algorithm to be aware of stability of the environment and update accordingly 
instead of updating solely based on the observed reward (plain EXP3).
"""
import numpy as np
from mab.algorithms.exp3_new import EXP3New
from MLO.utils import cal_tv_distance

class EXP3NewExperimental(EXP3New):
    name = "exp3_experimental"
    def __init__(self, n_arms, **kwargs):
        super().__init__(n_arms=n_arms, **kwargs)

    def cal_probs(self):
        "Compute the probability distribution over actions."
        # Normalize the log_weights to prevent numerical instability
        min_weight = np.min(self.weights)
        weights_normalized = self.weights - min_weight
        
        # mix with uniform
        weights = np.exp(-self.eta * weights_normalized)
        probs = weights / sum(weights)

        weight = 1 / (1 + cal_tv_distance(self.probs, probs))

        probs = (1-weight) * self.probs + weight * probs
        return probs / sum(probs)