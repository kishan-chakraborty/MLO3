"Experiment to determine Nash convergence for two players with different seed values. Parallel Computing"

from pathlib import Path
current_dir = Path(__file__).resolve().parent

import concurrent.futures
import numpy as np

from MLO.env import MABEnvironment
from mab.algorithms import exp3


import pickle

n_aps = 4
m_nonaps = 20
area_size = 10
d1 = 10 * np.sqrt(2)  # Ensure that all non-APs have at least one AP in visibility.
T = 200000
w0 = 32
m = 3
seeds = np.arange(1, 2, dtype=int)

data_dir = current_dir/'data'
data_dir.mkdir(exist_ok=True)


def run_experiment(seed):
    learner = exp3.EXP3
    kwargs = {}

    env = MABEnvironment(
        n_aps=n_aps,
        m_nonaps=m_nonaps,
        n_links=1,
        area_size=area_size,
        d1=d1,
        seed=int(seed),
        learner=learner,
        normalized=True,
        kwargs=kwargs
    )

    env.run(T, w0, m)
    file_path = str(data_dir) + f"/seed{seed}_{learner.name}_environment_data.pkl"

    with open(file_path, "wb") as f:
        pickle.dump(env, f)

    return f"Seed {seed} done"

with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(run_experiment, seeds))

print(results)
