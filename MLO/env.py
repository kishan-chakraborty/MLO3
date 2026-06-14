"""First experiment: MAB environment with multiple APs and non-APs using EXP3 learner."""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional

from nstr_model import ThroughputNSTR
from utils import AP, NonAP
from Algorithms.exp3 import EXP3

def euclid(a: np.ndarray, b: np.ndarray):
    "To find the distance between two AP and non-AP."
    return np.linalg.norm(a - b)


def build_arms_for_ap(ap_id: int, n_links: int) -> List[Tuple[int, int]]:
    "For a given AP, return the list of arms it offers."
    return [(ap_id, i_link) for i_link in range(n_links + 1)]


def compute_system_throughputs(aps: List[AP], w0, m, normalized) -> Dict[int, float]:
    """
    Calculate the system throughput for the current associations.
    Args:
        aps: List of AP objects.
        nonaps: List of NonAP objects.
        n_links: Number of links per AP (e.g., 2 for MLA with 2 links)
        associations: Dict mapping nonap_id -> (ap_id, arm_label)
        params: Dict of parameters including 'mla_scheme', 'payload', 'mla_params'
        normalized: Normalized throughput
    Returns:
        Dict mapping ap_id -> throughput in bits/s
    """
    # Build per-AP counts and call mla_models.mla_throughput
    ap_throughputs = {}

    for ap in aps:
        # We need to count no. of non_APs associated with single links and those connected with both links.
        [n_sld1, n_sld2, n_mld] = ap.n_association_per_link

        thr = ThroughputNSTR(n_sld1, n_sld2, n_mld, w0, m)
        sld_thr1, sld_thr2, mld_thr1, mld_thr2 = thr.cal_system_thr(normalized)

        ap_throughputs[ap.id] = [sld_thr1, sld_thr2, mld_thr1 + mld_thr2]
    return ap_throughputs

class MABEnvironment:
    def __init__(
        self,
        n_aps=3,
        m_nonaps=10,
        n_links=2,
        area_size=1.0,
        d1=0.4,
        seed: Optional[int] = None,
        learner=None,
        args=None,
        normalized=False,
    ):
        """
        n_aps: number of APs (each has two links)
        m_nonaps: number of non-AP devices (each an MLD)
        area_size: side length of square area
        d1: discovery radius (non-AP sees AP if within d1)
        params: simulation parameters for throughput etc.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.n_aps = n_aps  # No. of APs
        self.m_nonaps = m_nonaps  # No. of non-APs
        self.n_links = n_links  # No. of links per AP
        self.area = area_size  # Area size (square)
        self.d1 = d1  # Discovery radius
        self.throughputs = 0
        self.gamma = 0.1
        self.learner = learner
        self.learner_args = args # A dict to initialize the learner.
        self.normalized = normalized

        # place APs and non-APs
        self.aps: List[AP] = []
        self.nonaps: List[NonAP] = []
        self._init_nodes()  # Initialize positions and visibility

        # prepare learners for non-APs
        self._init_nonap_learners()

    def spawn_aps(self):
        # AP placement: uniformly random in area
        for i in range(self.n_aps):
            pos = (
                np.random.rand() * self.area,
                np.random.rand() * self.area,
            )  # Pos of the AP
            self.aps.append(AP(i, pos))
            self.n_association_per_link = [
                0 for _ in range(self.n_links + 1)
            ]  # Initialize association counts per link

    def spawn_nonaps(self):
        # non-AP placement
        for j in range(self.m_nonaps):
            pos = (
                np.random.rand() * self.area,
                np.random.rand() * self.area,
            )  # Pos of the non-AP
            self.nonaps.append(NonAP(j, pos))

    def _init_nodes(self):
        """Initialize positions of APs and non-APs, and compute visibility and available arms."""
        self.spawn_aps()
        self.spawn_nonaps()
        
        # compute visible APs and available arms
        for na in self.nonaps:
            na.visible_aps = [
                ap.id for ap in self.aps if euclid(na.pos, ap.pos) <= self.d1
            ]  # Visible APs.

            # build available arms as indices referencing the AP-specific arms in global_arm_list
            na.available_arms = []
            for ap_id in na.visible_aps:
                na.available_arms.extend(
                    build_arms_for_ap(ap_id, self.n_links)
                )  # All the arms {1, 2, {1, 2}} that the the non-AP can choose from.

    def _init_nonap_learners(self):
        "Assign learning mechanism (EXP3) to each non-AP based on available arms."
        for na in self.nonaps:
            K = len(na.available_arms)
            na.learner = self.learner(K, self.learner_args)

    def _update_ap_conn(self, na, act_idx):
        """
        If the non-ap choose a different connection, update connection"
        Args:
            na: non-ap
            chosen_arm: Current chosen arm
        """
        chosen_arm = na.available_arms[act_idx]  # (ap_id, arm_label)

        if len(na.chosen_arms) > 0:  # If there is a previous connection
            # Remove the previous AP connection
            last_chosen_arm = na.available_arms[na.chosen_arms[-1]]
            prev_ap_id, prev_ap_link = last_chosen_arm
            prev_ap = self.aps[prev_ap_id]
            prev_ap.associations.pop(na.id)
            prev_ap.n_association_per_link[prev_ap_link] -= 1

        # Update the association of the non-AP to the chosen AP and link.
        chosen_ap_id, chosen_link = chosen_arm
        chosen_ap = self.aps[chosen_ap_id]
        chosen_ap.associations[na.id] = chosen_link
        chosen_ap.n_association_per_link[chosen_link] += 1

        na.chosen_arms.append(act_idx)  # Store the index of the chosen arm

    def _update_learner_weights(self, na, ap_throughputs):
        "Update the learner's weight for each non-ap"
        chosen_arm_idx = na.chosen_arms[-1]
        connected_ap_id, connected_link = na.available_arms[chosen_arm_idx]
        reward = ap_throughputs[connected_ap_id][connected_link]
        action_idx = na.available_arms.index((connected_ap_id, connected_link))
        na.learner.update(action_idx, reward)
        na.throughputs.append(reward)
        self.throughputs += reward

    def step(self, w0, m):
        """
        Perform one time slot:
        - Each non-AP chooses an arm (if any)
        - Build associations mapping nonap_id -> (ap_id, arm)
        - Compute per-AP throughput (bps) and per-nonAP reward (0..1)
        - Update learners and return (ap_throughputs, total_system_reward)
        """
        # Select arms for each non-AP and update the association map.
        for na in self.nonaps:
            act_idx = na.learner.choose_action()  # Arm selected by the non-AP
            self._update_ap_conn(na, act_idx)

        # 2) compute the system throughputs per APs. ap_id -> [th1, th2, th12]
        ap_throughputs = compute_system_throughputs(self.aps, w0, m, self.normalized)

        # Update the learner weights for each non-AP
        for na in self.nonaps:
            if na.learner is None or na.chosen_arms is None:
                continue
            self._update_learner_weights(na, ap_throughputs)

    def run(self, T: int, w0: int, m: int):
        for iter in range(T):
            self.step(w0, m)

            # if iter % 10 == 0:
            #     print("No. of iter:", iter)