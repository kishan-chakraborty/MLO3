# Calculate per step throughput for adversarial env.
import numpy as np
import copy
from typing import List
from MLO.utils import AP, NonAP
from MLO.nstr import params
from MLO.n_player_nash_files import build_thr_table

import pickle
import matplotlib.pyplot as plt


def cal_joint_action(non_aps: List[NonAP], n_aps, n_actions, iter: int) -> List[tuple]:
    """
    Calculate the joint action of all devices at a given iteration, excluding the foc device.
    Args:
        non_aps: List of NonAP objects containing the history of chosen arms.
        n_aps: Total number of APs.
        n_actions: Number of available actions (per AP).
        iter: The current iteration for which to calculate the joint action.
        dev_id: The identity of the foc device to exclude from the joint action calculation.
    Returns:
        List of tuples representing the joint action wrt each AP and link [(3D tuple)xn_aps].
    """
    # Find the current joint actionn of all devices
    joint_actions = [[0] * n_actions for _ in range(n_aps)] # placeholder for joint action counts
    for n_ap in non_aps:        
        curr_act = n_ap.chosen_arms[iter]
        ass_ap, ass_link = n_ap.available_arms[curr_act]

        joint_actions[ass_ap][ass_link] += 1

    return joint_actions

def cal_per_step_regret(
    dev_id: int,
    n_aps: int,
    non_aps: List[NonAP],
    thr_tab: np.ndarray
) -> np.ndarray:
    """
    Calculate per step cumulative regret for a single foc AP.

    Parameters
    ----------
    dev_id: Identity of the device for which we want to calculate regret.
    n_aps: No. of APs
    non_aps: List of NonAP objects containing the history of chosen arms and throughputs
    thr_tab: Prebuilt table of throughputs

    Returns
    -------
    regret : np.ndarray, shape (T,)
        Cumulative regret up to each timestep.
    """
    foc_dev = non_aps[dev_id]
    foc_act = foc_dev.chosen_arms
    foc_probs = foc_dev.learner.save_probs
    foc_thr = foc_dev.throughputs
    n_actions = len(foc_dev.available_arms) # Available actions to a UE
    n_arms = 3 # Actions per AP

    alt_act_cumsum = np.zeros(n_actions)
    cum_avg_thr = 0
    regret_hist = np.zeros_like(foc_act, dtype=float)
    
    for iter, cur_act in enumerate(foc_act):
        curr_joint_actions = cal_joint_action(non_aps, n_aps, n_arms, iter)
        
        # Calculate alternate throughputs for choosing a different arm (from prebuilt table)
        cur_ap, cur_link = foc_dev.available_arms[cur_act]
        alt_thrs = np.zeros(n_actions, dtype=float)
        alt_joint_act = copy.deepcopy(curr_joint_actions)
        alt_joint_act[cur_ap][cur_link] -= 1

        for idx, (ap, link) in enumerate(foc_dev.available_arms):
            alt_joint_act_temp = copy.deepcopy(alt_joint_act)
            if (ap, link) == foc_dev.available_arms[cur_act]:
                alt_thrs[idx] = foc_thr[iter]
            else:
                alt_joint_act_temp[ap][link] += 1  # Increment the count for the alternate action

                # Get the alternate throughput from the prebuilt table
                alt_thrs[idx] = thr_tab[tuple(alt_joint_act_temp[ap])][link]

        alt_act_cumsum += alt_thrs
        cum_avg_thr += np.dot(foc_probs[iter], alt_thrs)

        regret_hist[iter] = np.max(alt_act_cumsum) - cum_avg_thr

    return regret_hist

if __name__ == "__main__":
    with open('ExperimentData/Experiment1/seed1_exp3++_environment_data.pkl', 'rb') as f:
        env = pickle.load(f)

    non_aps = env.nonaps
    aps = env.aps

    # Calculate thrs for different joint actions
    N, w0, m = 20, 32, 3
    thr_table = build_thr_table(N, w0, m, params, True)

    regret_hist = cal_per_step_regret(0, 4, non_aps, thr_table)

    plt.plot(regret_hist)
    plt.show()