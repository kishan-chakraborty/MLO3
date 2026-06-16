# bianchi.py
"""
Bianchi-style functions for saturated IEEE 802.11 DCF.
References:
 - Bianchi G., "Performance analysis of the IEEE 802.11 Distributed Coordination Function", IEEE J. Sel. Areas Commun., Mar 2000.
   (used for closed-form expression of tau given collision prob p and slot durations).
 - The Korolev et al. model uses Bianchi's RN values for networks of various sizes. See Korolev et al., (Equations referenced in other files). :contentReference[oaicite:5]{index=5}
"""

import numpy as np
from math import comb
from scipy.optimize import bisect
import matplotlib.pyplot as plt

params = {
    "channel_rate": 1_000_000.0,  # 1 Mbit/s
    "prop_delay": 1e-6,
    "slot_time": 50e-6,  # 50 us
    "SIFS": 28e-6,  # 28 us
    "DIFS": 128e-6,  # 128 us
    "phy_header_bits": 128,  # PHY header bits (FHSS)
    "mac_header_bits": 272,  # MAC header bits
    "rts_bits": 160,  # RTS bits (table shows '160 bits + PHY header')
    "cts_bits": 112,  # CTS bits (112 + PHY)
    "ack_bits": 112,  # ACK bits (112 + PHY)
    "payload_bits": 8184,  # payload (paper uses 8184 bits)
    "ACK_timeout": 300e-6,  # 300 us (sim only)
    "CTS_timeout": 300e-6,  # 300 us (sim only)
}


def tx_time(bits, r):
    return bits / r


def times_basic(params):
    r = params["channel_rate"]
    phy = tx_time(params["phy_header_bits"], r)
    mac = tx_time(params["mac_header_bits"], r)
    payload = tx_time(params["payload_bits"], r)

    # include PHY header with ACK as table shows "ACK = 112 bits + PHY header"
    T_s = (
        phy + mac + payload + params["SIFS"] + 2 * params["prop_delay"] + params["DIFS"]
    )
    T_c = phy + mac + payload + params["DIFS"] + params["prop_delay"]

    return T_s, T_c


def solve_tau_p(n, W, m, tol=1e-18, max_iter=2000):
    tau = 0.5
    for _ in range(max_iter):
        p = 1 - (1 - tau) ** (n - 1)  # Eq. (9)

        denom = 1 + W
        if abs(p - 1 / 2) > 1e-18:  # For p != 1/2
            denom += p * W * (((2 * p) ** m) - 1) / (2 * p - 1)

        tau_new = 2 / denom
        if abs(tau_new - tau) < tol:
            return tau, p
        tau = tau_new

    return tau, p


# Utility: precompute RN for a range of N
def calculate_throughput(n, w0, m, params):
    # If n == 0 then thr is zero
    if n == 0:
        return 0, 0, 0
    tau, p = solve_tau_p(n, w0, m)

    T_s, T_c = times_basic(params)
    P_tr = 1 - (1 - tau) ** n

    slot_time = params["slot_time"]
    payload_bits = params["payload_bits"]

    if P_tr <= 0:
        return 0.0, 0.0, 0.0
    P_s = (n * tau * (1 - tau) ** (n - 1)) / P_tr
    avg_slot_time = (1 - P_tr) * slot_time + P_tr * P_s * T_s + P_tr * (1 - P_s) * T_c
    thr = (P_tr * P_s * payload_bits) / avg_slot_time
    return thr * 1e-6, tau, p


if __name__ == "__main__":
    max_devices = 50
    throughputs = {i: 0.0 for i in range(1, max_devices + 1)}
    w0 = 128
    m = 3

    for n_devices in range(1, max_devices + 1):
        thr, _, _ = calculate_throughput(n_devices, w0, m, params)
        throughputs[n_devices] = thr

    plt.plot(list(throughputs.keys()), list(throughputs.values()))
    plt.show()
