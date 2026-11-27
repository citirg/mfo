"""Example: maximize WSR of a small MC-NOMA system with MFO."""
from mfo import mfo
from noma import MCNOMA

if __name__ == "__main__":
    for p_max in [0.2, 0.5, 1.0]:
        sys = MCNOMA(K=4, N=4, M=2, P_max=p_max, seed=0)
        x, f, hist = mfo(sys.wsr, sys.dim, 0.0, 1.0, pop_size=30, iters=50, seed=1)
        print(f"P_max={p_max:.1f}  WSR={f:.4e}  (iter 1: {hist[0]:.4e})")
