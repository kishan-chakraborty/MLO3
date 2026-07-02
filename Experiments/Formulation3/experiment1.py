"""
In this formulation, the primary goal is to combine the experimentations of the last two
formulations. First, we start with a each UEs selecting an AP rather than AP-link combinations
for a fixed duration. Followed by this, the UEs associated with their respective AP select the 
optimal link from their associated AP.
"""
"Experiment to determine Nash convergence for two players with different seed values. Parallel Computing"

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from MLO.env import TestBed, Experiment
from MLO.utils import AP, NonAP
from MLO.nstr import params
from MLO import n_player_nash_files, regret
from MLO.n_player_nash_files import build_thr_table
from MLO.regret import cal_per_step_regret

from Algorithms.exp3 import EXP3

import pickle
from MLO.env import build_arms_for_ap


n_aps = 4
m_nonaps = 20
area_size = 10
d1 = 10 * np.sqrt(2)  # Ensure that all non-APs have at least one AP in visibility.

# The horizon is divided into two parts (T1 and T2).
n_links = 2 # No of links available per AP.
horizon = 200_000
T1 = 80_000
T2 = horizon-T1
w0 = 32
m = 3
seed = 1

# data_dir = current_dir/'data'
# data_dir.mkdir(exist_ok=True)
"""
1. It will be a two stage experiment.
2. First associate with the best AP.
3. Within that AP, choose the best link.
"""

# Create the test bed with just one link per AP (Let each non AP associate with an AP first.)
test_bed = TestBed(n_aps=n_aps, m_nonaps=m_nonaps, n_links=1, area_size=area_size, d1=d1, seed=seed)

aps = test_bed.aps
non_aps = test_bed.nonaps

# Assign learners to the non_aps.
learner = EXP3
for id, non_ap in enumerate(non_aps):
    n_actions = len(non_ap.available_actions)
    args = {"gamma": 0.1, "seed":id+1}
    non_ap.learner = learner(n_actions, **args)

args = {'w0': w0, 'm': m, 'store_rewards': True}
experiment = Experiment(aps, non_aps, T1, 1, **args)
experiment.run()

# Copy the APs and Non APs to store data
aps2 = []
non_aps2 = []
for i in range(n_aps):
    ap = AP(aps[i].id, aps[i].pos)
    aps2.append(ap)

for i in range(m_nonaps):
    non_ap = NonAP(non_aps[i].id, non_aps[i].pos)
    non_aps2.append(non_ap)

for ap in aps2:
    ap.associations = {}
    ap.n_association_per_link = [0] * 3

for id, non_ap in enumerate(non_aps2):
    highest_prob_idx = np.argmax(non_aps[id].learner.probs)
    assc_ap_id = non_aps[id].available_actions[highest_prob_idx][0]

    # Make the associated AP the only available AP and build the arm space accordingly.
    non_ap.visible_aps = [assc_ap_id]
    non_ap.curr_act_id = None
    non_ap.available_actions = build_arms_for_ap(assc_ap_id, n_links)

    # Update the learner.
    n_actions = len(non_ap.available_actions)
    args = {"gamma": 0.1, "seed":id+1}
    non_ap.learner = learner(n_actions, **args)

args = {'w0': w0, 'm': m, 'store_rewards': True}
experiment = Experiment(aps2, non_aps2, T2, 1, **args)
experiment.run()

# Expand the arm space corresponding to the UEs.
new_arm_space = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
arm_space_len = len(new_arm_space)

for non_ap in non_aps:
    new_arm_idx = [new_arm_space.index(arm) for arm in non_ap.available_actions]
    for t in range(T1):
        # Chosen arm by the non_ap at t (old arm space)
        id = non_ap.chosen_arms[t]
        id_new = new_arm_idx[id]
        non_ap.chosen_arms[t] = id_new

        # Store the probs corresponding to new arm index.
        new_probs = np.zeros(arm_space_len)
        
        for i, prob in enumerate(non_ap.learner.save_probs[t]):
            new_idx = new_arm_idx[i]
            new_probs[new_idx] = prob

        non_ap.learner.save_probs[t] = new_probs

    non_ap.available_actions = new_arm_space

for non_ap in non_aps2:
    new_arm_idx = [new_arm_space.index(arm) for arm in non_ap.available_actions]
    for t in range(T1):
        # Chosen arm by the non_ap at t (old arm space)
        id = non_ap.chosen_arms[t]
        id_new = new_arm_idx[id]
        non_ap.chosen_arms[t] = id_new

        # Store the probs corresponding to new arm index.
        new_probs = np.zeros(arm_space_len)
        
        for i, prob in enumerate(non_ap.learner.save_probs[t]):
            new_idx = new_arm_idx[i]
            new_probs[new_idx] = prob

        non_ap.learner.save_probs[t] = new_probs

    non_ap.available_actions = new_arm_space

# Calculating regret
N, w0, m = 20, 32, 3
thr_table = build_thr_table(N, w0, m, params, True)

for non_ap1, non_ap2 in zip(non_aps, non_aps2):
    non_ap1.chosen_arms.extend(non_ap2.chosen_arms)
    non_ap1.rewards.extend(non_ap2.rewards)
    non_ap1.learner.save_probs.extend(non_ap2.learner.save_probs)

regret_hist1 = cal_per_step_regret(0, 4, non_aps, thr_table)
regret_hist2 = cal_per_step_regret(0, 4, non_aps2, thr_table)

sqrt_regret1 = np.sqrt(np.arange(1, len(regret_hist1)+1))
sqrt_regret2 = np.sqrt(np.arange(1, len(regret_hist2)+1))

plt.plot(regret_hist1, label='Per-step Regret (multi player)')
plt.plot(regret_hist2, label='Per-step Regret (multi player)')
# plt.plot(regret_hist2, label='Per-step Regret (multi player)')
# plt.plot(5.5*sqrt_regret_pp, label='sqrt(t)', linestyle='--')
plt.legend()
plt.show()
