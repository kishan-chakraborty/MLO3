"""
Testing the new environment.
"""
from pathlib import Path
current_dir = Path(__file__).resolve().parent

import numpy as np
import concurrent.futures

from MLO.env import TestBed, Experiment
from Algorithms.exp3 import EXP3

import pickle

n_aps = 4
m_nonaps = 20
area_size = 10
d1 = 10 * np.sqrt(2)  # Ensure that all non-APs have at least one AP in visibility.

data_dir = current_dir/'data'
data_dir.mkdir(exist_ok=True)

seeds = np.arange(1, 2, dtype=int)

horizon = 100_000
w0 = 32
m = 3

def run_experiment(seed):
    # Create the test bed
    test_bed = TestBed(n_aps=n_aps, m_nonaps=m_nonaps, n_links=2, area_size=area_size, d1=d1, seed=seed)

    non_aps = test_bed.nonaps

    # Assign learners to the non_aps.
    learner = EXP3
    for id, non_ap in enumerate(non_aps):
        n_actions = len(non_ap.available_actions)
        args = {"gamma": 0.1, "seed":id+1}
        non_ap.learner = learner(n_actions, **args)

    args = {'w0': w0, 'm': m}
    experiment = Experiment(test_bed, horizon, 1, **args)
    experiment.run()

    file_path = str(data_dir) + f"/seed{seed}_{learner.name}_environment_data.pkl"

    with open(file_path, "wb") as f:
        pickle.dump(experiment, f)

    return f"Seed {seed} done"

with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(run_experiment, seeds))

print(results)