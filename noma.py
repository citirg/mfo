"""Minimal downlink multicarrier NOMA weighted-sum-rate (WSR) model.

K users share N subcarriers; at most M users are superposed per subcarrier
and decoded with successive interference cancellation (SIC) in descending
order of channel gain. A candidate solution is a vector in [0, 1]^(K*N)
that is mapped to a feasible power allocation (top-M users per subcarrier,
total power scaled to P_max).
"""
import numpy as np


class MCNOMA:
    def __init__(self, K=4, N=4, M=2, P_max=1.0, B=1e6, noise=1e-9,
                 weights=None, seed=0):
        rng = np.random.default_rng(seed)
        self.K, self.N, self.M, self.P_max, self.B = K, N, M, P_max, B
        self.noise = noise
        self.w = np.ones(K) if weights is None else np.asarray(weights, float)
        # Rayleigh fading with distance-based path loss
        dist = rng.uniform(50, 300, size=K)
        pl = 1e-3 * dist ** -3.0
        self.g = pl[:, None] * rng.exponential(1.0, size=(K, N))  # K x N
        self.dim = K * N

    def allocate(self, x):
        """Map a raw vector in [0, 1]^(K*N) to a feasible power matrix."""
        P = np.clip(np.asarray(x, float).reshape(self.K, self.N), 0, None)
        for n in range(self.N):
            keep = np.argsort(P[:, n])[-self.M:]
            mask = np.zeros(self.K, bool)
            mask[keep] = True
            P[~mask, n] = 0.0
        s = P.sum()
        return P * (self.P_max / s) if s > 0 else P

    def wsr(self, x):
        P = self.allocate(x)
        rate = np.zeros(self.K)
        for n in range(self.N):
            g, p = self.g[:, n], P[:, n]
            for k in np.flatnonzero(p > 0):
                # interference from users with stronger channels (not cancelled)
                stronger = (g > g[k]) & (p > 0)
                interf = g[k] * p[stronger].sum()
                sinr = g[k] * p[k] / (interf + self.noise)
                rate[k] += self.B / self.N * np.log2(1 + sinr)
        return float((self.w * rate).sum())
