import numpy as np
from nstr import ThroughputNSTR
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
        self.n_association_per_link = [0, 0, 0]


class NonAP:
    def __init__(self, nonap_id: int, pos: Tuple[float, float]):
        self.id = nonap_id
        self.pos = np.array(pos)  # Position of the Non-AP
        self.visible_aps: List[int] = []  # AP ids within radius d1 (populated by env)
        self.available_arms: List[Tuple[int, int]] = []  # list of (ap_id, arm_label)
        self.learner = None
        self.chosen_arms: List[tuple] = []  # Record all the cosen arms.
        self.throughputs: List[float] = []  # Record the throughputs per step.