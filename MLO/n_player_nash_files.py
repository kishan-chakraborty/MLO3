import numpy as np
from MLO.nstr import ThroughputNSTR

NUM_LINKS = 3   # link0, link1, both — fixed by the system model


# -----------------------------------------------------------------------------
# Section 1: Lookup table construction
# -----------------------------------------------------------------------------

def get_valid_configs(N: int) -> list[tuple[int, int, int]]:
    """
    Enumerate all unique configs (x, y, z) with x + y + z <= N and x >= y.

    The x >= y restriction exploits the link0/link1 symmetry so each unique
    config is computed only once. The mirror (y, x, z) is filled in
    build_thr_table().

    Parameters
    ----------
    N : int
        Maximum number of devices that can associate with an AP.

    Returns
    -------
    list of (x, y, z) tuples — unique configs to compute.
    """
    configs = []
    for x in range(N + 1):
        for y in range(x + 1):                 # y <= x enforces symmetry
            for z in range(N + 1 - x - y):     # x + y + z <= N
                configs.append((x, y, z))
    return configs


def build_thr_table(N: int, w0: float, m: float, params, normalized=False) -> np.ndarray:
    """
    Build the (N+1) x (N+1) x (N+1) x NUM_LINKS throughput lookup table.

    Each entry thr_table[x, y, z] is a 3-element array:
        [sld1, sld2, mld1 + mld2]

    Only configs with x + y + z <= N are computed; the rest remain 0.
    Symmetry thr_table[x, y, z] == thr_table[y, x, z] is exploited to
    halve the number of ThroughputNSTR solves.

    Parameters
    ----------
    N : int
        Maximum number of devices per AP.
    w0 : float
        System parameter passed to ThroughputNSTR.
    m : float
        System parameter passed to ThroughputNSTR.

    Returns
    -------
    thr_table : np.ndarray, shape (N+1, N+1, N+1, NUM_LINKS)
        Lookup table indexed as thr_table[x, y, z, chosen_link].
    """
    thr_table = np.zeros((N + 1, N + 1, N + 1, NUM_LINKS), dtype=float)

    for (x, y, z) in get_valid_configs(N):
        thr = ThroughputNSTR(x, y, z, w0, m, params)
        thr.solve_s_p()

        sld1, sld2, mld1, mld2 = thr.cal_system_thr(normalized=normalized)

        # [link0 thr, link1 thr, both thr]
        throughputs = (sld1, sld2, mld1 + mld2)

        # Fill primary entry (x >= y)
        thr_table[x, y, z] = throughputs

        # Mirror symmetric entry (swap x <-> y, swap sld1 <-> sld2)
        if x != y:
            mirrored = np.array([sld2, sld1, mld1 + mld2], dtype=float)
            thr_table[y, x, z] = mirrored

    return thr_table


# -----------------------------------------------------------------------------
# Section 2: Config helpers
# -----------------------------------------------------------------------------

def make_config(joint_actions: list[int]) -> tuple[int, int, int]:
    """
    Convert a list of per-AP actions into a joint config tuple (x, y, z).

    Parameters
    ----------
    joint_actions : list of ints, each in {0, 1, 2}
        Action chosen by each AP at a given timestep.

    Returns
    -------
    (x, y, z) : tuple
        Number of APs that chose link0, link1, and both respectively.

    Example
    -------
    >>> make_config([0, 1, 0, 2])
    (2, 1, 1)
    """
    counts = [0] * NUM_LINKS
    for a in joint_actions:
        counts[a] += 1
    return tuple(counts)


# -----------------------------------------------------------------------------
# Section 3: Throughput lookup
# -----------------------------------------------------------------------------

def cal_thr(
    chosen_link: int,
    config: tuple[int, int, int],
    thr_table: np.ndarray,
) -> float:
    """
    Look up the throughput for one AP given its chosen link and joint config.

    Parameters
    ----------
    chosen_link : int
        Action chosen by this AP: 0 = link0, 1 = link1, 2 = both.
    config : tuple (x, y, z)
        Joint config — counts of APs on each link.
    thr_table : np.ndarray, shape (N+1, N+1, N+1, NUM_LINKS)
        Prebuilt lookup table from build_thr_table().

    Returns
    -------
    float
        Throughput for this AP under the given config and chosen link.
    """
    x, y, z = config
    return thr_table[x, y, z, chosen_link]


# -----------------------------------------------------------------------------
# Section 4: Alternative throughput calculation
# -----------------------------------------------------------------------------

def cal_alt_thr(
    joint_actions: list[int],
    focal_ap_idx: int,
    thr_table: np.ndarray,
) -> list[float]:
    """
    Compute the throughput the focal AP would receive under each alternative
    action, holding all other APs' actions fixed.

    Parameters
    ----------
    joint_actions : list of ints
        Current action of every AP.
    focal_ap_idx : int
        Index of the AP whose alternatives we are evaluating.
    thr_table : np.ndarray, shape (N+1, N+1, N+1, NUM_LINKS)
        Prebuilt lookup table from build_thr_table().

    Returns
    -------
    alt_thrs : list of floats, length NUM_LINKS
        alt_thrs[a] = throughput focal AP would get if it chose action a,
        with all other APs keeping their current actions.
    """
    alt_thrs = []
    for alt_action in range(NUM_LINKS):
        # Build counterfactual joint action with focal AP switching to alt_action
        counterfactual = list(joint_actions)
        counterfactual[focal_ap_idx] = alt_action
        cf_config = make_config(counterfactual)
        alt_thrs.append(cal_thr(alt_action, cf_config, thr_table))
    return alt_thrs


# -----------------------------------------------------------------------------
# Section 5: System exploitability (Nash gap)
# -----------------------------------------------------------------------------

def cal_joint_act(
    dev_acts: list[np.ndarray],
    dev_thrs: list[np.ndarray],
    thr_table: np.ndarray,
) -> tuple[list[float], np.ndarray]:
    """
    Compute per-step system exploitability for n APs.

    At each timestep t, the exploitability is the maximum gain any single AP
    could achieve by deviating unilaterally from the current joint action.

    Parameters
    ----------
    dev_acts : list of 1-D int arrays, one per AP, each shape (T,)
        dev_acts[i][t] = action chosen by AP i at step t.
    dev_thrs : list of 1-D float arrays, one per AP, each shape (T,)
        dev_thrs[i][t] = throughput received by AP i at step t.
    thr_table : np.ndarray, shape (N+1, N+1, N+1, NUM_LINKS)
        Prebuilt lookup table from build_thr_table().

    Returns
    -------
    last_alt_thrs : list of floats, length NUM_LINKS
        Alternative throughputs for the most-deviating AP at the final step.
    system_exploitability : np.ndarray, shape (T,)
        Per-step system exploitability.
    """
    n_players = len(dev_acts)
    T = len(dev_acts[0])
    system_exploitability = np.zeros(T, dtype=float)
    last_alt_thrs = None

    for t in range(T):
        joint_actions = [dev_acts[i][t] for i in range(n_players)]
        max_gain = 0.0

        for ap_idx in range(n_players):
            cur_thr = dev_thrs[ap_idx][t]
            alt_thrs = cal_alt_thr(joint_actions, ap_idx, thr_table)
            gain = max(alt_thrs) - cur_thr

            if gain > max_gain:
                max_gain = gain
                last_alt_thrs = alt_thrs

        system_exploitability[t] = max_gain

    return last_alt_thrs, system_exploitability


# -----------------------------------------------------------------------------
# Section 6: Per-step regret (Exp3 / online learning)
# -----------------------------------------------------------------------------

def cal_per_step_regret(
    dev_acts: list[np.ndarray],
    focal_ap_idx: int,
    focal_thr: np.ndarray,
    focal_probs: np.ndarray,
    thr_table: np.ndarray,
) -> np.ndarray:
    """
    Calculate per-step cumulative regret for a single focal AP.

    Regret at step t = (best cumulative reward in hindsight up to t)
                     - (cumulative expected reward up to t)

    Parameters
    ----------
    dev_acts : list of 1-D int arrays, one per AP, each shape (T,)
        dev_acts[i][t] = action chosen by AP i at step t.
    focal_ap_idx : int
        Index of the AP whose regret we are computing.
    focal_thr : 1-D float array, shape (T,)
        Observed throughput of the focal AP at each step.
    focal_probs : 2-D float array, shape (T, NUM_LINKS)
        Probability distribution over actions used by the focal AP at each step.
    thr_table : np.ndarray, shape (N+1, N+1, N+1, NUM_LINKS)
        Prebuilt lookup table from build_thr_table().

    Returns
    -------
    regret : np.ndarray, shape (T,)
        Cumulative regret up to each timestep.
    """
    T = len(focal_thr)
    regret = np.zeros(T, dtype=float)

    cum_reward_per_act = np.zeros(NUM_LINKS, dtype=float)
    cum_expected_thr = 0.0

    for t in range(T):
        joint_actions = [dev_acts[i][t] for i in range(len(dev_acts))]
        focal_action = joint_actions[focal_ap_idx]

        # Throughput for each alternative action at this step
        alt_thrs = cal_alt_thr(joint_actions, focal_ap_idx, thr_table)

        # Replace focal AP's slot with the actual observed throughput
        alt_thrs[focal_action] = focal_thr[t]

        # Accumulate reward per action (hindsight best arm)
        cum_reward_per_act += np.array(alt_thrs)
        max_alt_thr = np.max(cum_reward_per_act)

        # Accumulate expected throughput under the played mixed strategy
        expected_thr = np.dot(alt_thrs, focal_probs[t])
        cum_expected_thr += expected_thr

        regret[t] = max_alt_thr - cum_expected_thr

    return regret


# -----------------------------------------------------------------------------
# Section 7: Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Replace this stub with your actual ThroughputNSTR import, e.g.:
    #   from throughput_nstr import ThroughputNSTR
    #
    # The class must implement:
    #   __init__(self, x, y, z, w0, m)
    #   solve_s_p(self)
    #   cal_system_thr(self, normalized=True) -> (sld1, sld2, mld1, mld2)
    # ------------------------------------------------------------------
    """Manual entries to verify the correctness of the result"""
    TWO_PLAYER_THR = {
        (2,0,0): (0.4344506928579139, 0.0,                0.0,),
        (0,2,0): (0.0,                0.4344506928579139, 0.0),
        (0,0,2): (0.0,                0.0,                0.8575426970492218,),
        (1,1,0): (0.8599348534201959, 0.8599348534201959, 0.0,),
        (1,0,1): (0.8227169848026108, 0.0,                0.8369479523809082),
        (0,1,1): (0.0,                0.8227169848026108, 0.8369479523809082),
    }

    # Build lookup table for N=2
    N, w0, m = 2, 32, 3
    thr_table = build_thr_table(N, w0, m)

    # Simulate 2 APs choosing actions randomly over T steps
    T = 10
    np.random.seed(42)
    n_players = 2
    dev_acts = [np.random.randint(0, NUM_LINKS, T) for _ in range(n_players)]

    # Derive observed throughputs from the table
    dev_thrs = []
    for i in range(n_players):
        thrs = [
            cal_thr(
                dev_acts[i][t],
                make_config([dev_acts[j][t] for j in range(n_players)]),
                thr_table,
            )
            for t in range(T)
        ]
        dev_thrs.append(np.array(thrs))

    print(dev_thrs)

    # # Exploitability
    # last_alts, exploitability = cal_joint_act(dev_acts, dev_thrs, thr_table)
    # print(f"\nPer-step exploitability:\n{exploitability}")

    # Regret for AP 0 (uniform mixed strategy)
    # focal_probs = np.ones((T, NUM_LINKS)) / NUM_LINKS
    # regret = cal_per_step_regret(
    #     dev_acts,
    #     focal_ap_idx=0,
    #     focal_thr=dev_thrs[0],
    #     focal_probs=focal_probs,
    #     thr_table=thr_table,
    # )
    # print(f"\nPer-step cumulative regret (AP 0):\n{regret}")
