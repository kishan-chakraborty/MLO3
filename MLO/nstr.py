"""
Implementation of the Korolev et al. NSTR analytical model using:
For every resource (single link access, multiple link access) there will be a cost function (reward function) based on load.
 - Equations (1) and (2): per-STA throughput for legacy STA and MLD device throughput.
 - Equations (3) and (4): average virtual slot time T_v and s_ij shares.
 - Equation (5): pi_X,Y via conditional binomial-like probabilities.

References:
  - Korolev N., Levitsky I., Khorov E., "Analytical Model of Multi-Link Operation in Saturated Heterogeneous Wi-Fi 7 Networks", IEEE WCL, 2022.
    See Equations (1)-(5) for direct formulae. :contentReference[oaicite:6]{index=6}
  - Bianchi (2000) for RN calculation and tau values used in conditional probs. :contentReference[oaicite:7]{index=7}
"""
import numpy as np
from math import comb
from typing import Dict, Tuple

from .bianchi import calculate_throughput, times_basic

params = {
    "channel_rate": 216_000_000.0,  # 216 Mbit/s
    "prop_delay": 1e-6,
    "slot_time": 9e-6,  # 9 us
    "SIFS": 16e-6,  # 16 us
    "DIFS": 34e-6,  # 34 us
    "phy_header_bits": 128,  # PHY header bits (FHSS)
    "mac_header_bits": 272,  # MAC header bits
    "payload_bits": 8184,  # payload (paper uses 8184 bits)
}

class ThroughputNSTR:
    def __init__(self, n_sld1, n_sld2, n_mld, w0, m, params=params):
        """
        Class to calculate throughput for given configuration.
        Args:
            n1_sld: No. of SLDs in link 1.
            n2_sld: No. of SLDs in link 2.
            n_mld: No. of MLDs in both the link.
            params: Other required paremeters.
        """
        self.n_sld1 = n_sld1
        self.n_sld2 = n_sld2
        self.n_mld = n_mld
        self.w0 = w0  # Minimum counter length.
        self.m = m  # No. of retries.
        self.params = params
        self.t_e = self.params["slot_time"]
        self.t_s, self.t_c = times_basic(self.params)

        # Calculate throughput for different combinations.
        max_devices = max(n_sld1, n_sld2) + n_mld
        min_devices = min(n_sld1, n_sld2)

        self.throughputs = {i: 0.0 for i in range(min_devices, max_devices + 1)}
        self.taus = {i: 0.0 for i in range(min_devices, max_devices + 1)}

        self._cal_per_sta_thr(min_devices, max_devices, self.w0, self.m)

        # Build P_XY_NiNj lookup table
        self.cond_p1 = self._build_P_XY_NiNj_tab(self.n_sld1)
        self.cond_p2 = self._build_P_XY_NiNj_tab(self.n_sld2)

        # Initialize s values for two  different channels.
        self.s1 = (1 / (self.n_mld + 1)) * np.ones(self.n_mld + 1)
        self.s2 = (1 / (self.n_mld + 1)) * np.ones(self.n_mld + 1)

        # Create the prob. tabels.
        self.t1 = None
        self.t2 = None
        self.p1 = np.zeros((self.n_sld1 + 1, self.n_mld + 1))
        self.p2 = np.zeros((self.n_sld2 + 1, self.n_mld + 1))

    def _cal_per_sta_thr(self, min_device, max_device, w0, m):
        """
        Build lookup table per device.
        Args:
            min_device: Minimmum no. of associated devices.
            max_device: Maximmum no. of associated devices.
            w0: Minimum counter length.
            m: No. of retries.
        Return:
            Lookup table for tau
        """
        for n in range(min_device, max_device + 1):
            thr, tau, _ = calculate_throughput(n, w0, m, self.params)
            self.throughputs[n] = thr
            self.taus[n] = tau

    def _build_P_XY_NiNj_tab(self, n_sld):
        """
        Build a lookup table for P_XY_NiNj for various config of dim  n_sld x n_mld x n_mld.

        """
        res = np.zeros((n_sld + 1, self.n_mld + 1, self.n_mld + 1))
        for k in range(self.n_mld + 1):
            for x in range(n_sld + 1):
                for y in range(k + 1):
                    res[x, y, k] = self.calculate_pi_x_y_n_k(n_sld, k, x, y)

        return res

    def calculate_pi_x_y_n_k(self, n_sld, n_mld, x, y):
        """
        CConditional probability that X legacy devices and Y MLD-affiliated STAs
        transmit in a given virtual slot, *given* that N legacy STAs and K MLD-affiliated STAs
        contend on this channel.
        Args:
            n_sld: No. of SLDs associated with the link.
            n_mld: No. of stations in the MLDs associated with the link at the moment.
            x: No. of legacy device transmitting.
            y: No. of MLDs transmitting.
        Return:
            probability
        """
        total = n_sld + n_mld
        c1 = comb(n_sld, x)
        c2 = comb(n_mld, y)

        res = (
            c1
            * c2
            * (self.taus[total] ** (x + y))
            * ((1 - self.taus[total]) ** (total - x - y))
        )
        return res

    def solve_s_p(
        self,
        max_iter: int = 200,
        tol: float = 1e-8,
    ):
        """
        Solve s and p iteratively.
        """
        for iter in range(max_iter):

            # Update the p values based on the values of s_ji's.
            self._update_p_values(self.n_sld1, self.s2, self.cond_p1, self.p1)
            self._update_p_values(self.n_sld2, self.s1, self.cond_p2, self.p2)

            # Update the t values
            self.t1 = self._update_t_values(self.p1)
            self.t2 = self._update_t_values(self.p2)

            # Update the s values.
            old_s1 = self.s1.copy()
            old_s2 = self.s2.copy()

            self._update_s_values(self.t1, self.p1, self.s1)
            self._update_s_values(self.t2, self.p2, self.s2)

            # See convergence of s
            err1 = max(abs(old_s1 - self.s1))
            err2 = max(abs(old_s2 - self.s2))

            max_error = max(err1, err2)

            # if iter % 10 == 0:
            #     print(f"Max error at iter: {iter} is {max_error}")

            if max_error < tol:
                break
        return self.s1, self.s2

    def _update_p_values(self, n_sld, s, cond_p, p):
        "Update the p values"
        for x in range(n_sld + 1):
            for y in range(self.n_mld + 1):
                p[x, y] = sum(s[::-1] * cond_p[x, y, :])

    def _update_s_values(self, t, p, s):
        "Update the values of s"
        n_sld, n_mld = p.shape
        if n_mld == 1:
            s[0] = 1
            return

        val = 0
        if n_sld > 1:  # No. SLD
            val = p[1, 0]

        # Update the s1 values.
        s[0] = (1 / t) * (
            p[0, 0] * self.t_e + val * self.t_s + self.t_c * sum(p[2:, 0])
        )

        s[1] = (1 / t) * (p[0, 1] * self.t_s + self.t_c * sum(p[1:, 1]))

        temp = self.t_c * p[:, 2:].sum(axis=0)
        s[2:] = temp * (1 / t)

    def _update_t_values(self, p):
        """
        Calculate the average duration of virtual slot.
        Args:
            n_sld: No. of SLDs
        """
        # Calcculating the value of T.
        n_slds, n_mlds = p.shape
        temp = 0
        if n_slds > 1:  # Zeros SLD
            temp += p[1, 0]
        if n_mlds > 1:
            temp += p[0, 1]

        t = p[0, 0] * self.t_e + temp * self.t_s

        # Calculate the summation
        total = sum(p.sum(axis=0)) - (p[0, 0] + temp)
        return t + total * self.t_c

    def cal_system_thr(self, normalized=False):
        _, _ = self.solve_s_p()
        # Calculate SLD throughputs
        sld_thr1, mld_thr1 = self._cal_link_thr(self.s2, self.n_sld1) 
        sld_thr2, mld_thr2 = self._cal_link_thr(self.s1, self.n_sld2)

        if normalized:
            channel_rate_mbps = self.params['channel_rate'] / (10**6)
            sld_thr1, mld_thr1 = sld_thr1/ channel_rate_mbps, mld_thr1/ channel_rate_mbps
            sld_thr2, mld_thr2 = sld_thr2/ channel_rate_mbps, mld_thr2/ channel_rate_mbps

        return sld_thr1, sld_thr2, mld_thr1, mld_thr2

    def _cal_link_thr(self, s, n_sld):
        """
        Calculate throughput corresponding to a link
        Args:
            s: s_ji table.
            n_sld: No. SLD associated with the link.
        return:
                Throughput for the link.
        """
        thrs = np.array(
            [self.throughputs[k] for k in range(n_sld, n_sld + self.n_mld + 1)]
        )
        num_sld = s[::-1] * thrs
        den_sld = np.arange(n_sld, n_sld + self.n_mld + 1)

        num_mld = num_sld * np.arange(self.n_mld + 1)

        if n_sld == 0:
            num_mld = num_mld[1:]
            den_sld = den_sld[1:]
            thr_sld = 0
        else:
            thr_sld = sum(num_sld / den_sld)

        if self.n_mld == 0:
            thr_mld = 0
        else:
            thr_mld = (1 / self.n_mld) * sum(num_mld / den_sld)

        return thr_sld, thr_mld


if __name__ == "__main__":
    n_sld1 = 0
    n_sld2 = 0
    n_mld = 2
    w0 = 32
    m = 3

    thr = ThroughputNSTR(n_sld1, n_sld2, n_mld, w0, m)

    # Calculate the throughput
    sld_thr1, sld_thr2, mld_thr1, mld_thr2 = thr.cal_system_thr(normalized=False)

    thr_per_device = [sld_thr1, sld_thr2, mld_thr1 + mld_thr2]
    print("Throughput of perdevice SLD and MLD", thr_per_device)
    print("System throughput is: ", sum(thr_per_device))
