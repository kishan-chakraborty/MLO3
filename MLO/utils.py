import numpy as np
from typing import List, Tuple, Dict
import seaborn as sns

pref_colors = ["#0b559f", "#2c944c", "#e05206", "#572c92", "#d51965", "#8f3371"]
NUM_LINKS = 3

class AP:
    def __init__(self, ap_id: int, pos: Tuple[float, float]):
        self.id = ap_id
        self.pos = np.array(pos)  # Position of the AP
        # For bookkeeping: which non-APs are currently associated.
        self.associations: Dict[int, int] = {}  # non-ap id -> connected link
        # Which link has how many devices connected. [n1_sld, n2_sld, n_mld]
        self.n_association_per_link = None


class NonAP:
    def __init__(self, nonap_id: int, pos: Tuple[float, float]):
        self.id = nonap_id
        self.pos = np.array(pos)  # Position of the Non-AP
        self.visible_aps: List[int] = []  # AP ids within radius d1 (populated by env)
        self.available_arms: List[Tuple[int, int]] = []  # list of (ap_id, arm_label)
        self.learner = None
        self.chosen_arms: List[tuple] = []  # Record all the cosen arms.
        self.throughputs: List[float] = []  # Record the throughputs per step.

def cal_kl_distance(dist1, dist2):
    """
    To implement the KL divergence change in probabilty distribution of two consecutive probabilities over 
    a set of actions. 
    dist1: Probability dist. over A. (say at time t) shape T-1 x n_arms
    dist2: Probability dist. over A. (say at time t+1), shape T-1 x n_arms
    """
    prob1 = np.array(dist1) # Prob at t
    prob2 = np.array(dist2) # Prob at t+1

    temp = np.log(prob2 / prob1)
    return np.sum(prob2 * temp, axis=1)

def cal_tv_distance(dist1, dist2):
    """
    Similar as above, calculate the total variation distance for two probability dist.
    """
    prob1 = np.array(dist1) # Prob at t
    prob2 = np.array(dist2) # Prob at t+1
    return 0.5 * np.abs(prob1-prob2).sum(axis=1)

def cal_log_ratio_distance(dist1, dist2):
    """
    Not a typical distance but log of exponential values will give some information.
    """
    return np.log(dist2) - np.log(dist1)