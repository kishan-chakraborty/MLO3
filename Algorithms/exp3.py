"""
This is a variation of EXP3 with the intention to do experimentation for congestion game. 
I want the algorithm to be aware of stability of the environment and update accordingly 
instead of updating solely based on the observed reward.
"""
import numpy as np
from mab.algorithms.exp3 import EXP3
from MLO.utils import cal_tv_distance

class EXP3Experimental(EXP3):
    name = "exp3_experimental"
    def __init__(self, n_arms, **kwargs):
        super().__init__(n_arms=n_arms, **kwargs)

    def cal_probs(self):
        "Compute the probability distribution over actions."
        # Normalize the log_weights to prevent numerical instability
        max_log_weight = np.max(self.log_weights)
        log_weights_normalized = self.log_weights - max_log_weight
        
        # mix with uniform
        weights = np.exp(log_weights_normalized)
        probs = (1 - self.gamma) * (weights / weights.sum()) + (
            self.gamma / self.n_arms
        )

        weight = 1 / (1 + cal_tv_distance(self.probs, probs))

        probs = (1-weight) * self.probs + weight * probs
        return probs / sum(probs)