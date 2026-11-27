"""Mudik Flow Optimization (MFO).

A population-based metaheuristic inspired by the collective movement of
travellers during the Indonesian "mudik" (homecoming) season. Each agent:

  1. moves toward the destination (the global best solution),
  2. follows the group flow (the population mean), and
  3. avoids congestion: in crowded regions it explores alternative routes.

Position update for agent i at iteration t:

    X_i <- X_i + (1 - C_i) * [ alpha*r1*(X_best - X_i) + beta*r2*(X_group - X_i) ]
               + C_i * gamma * r3 * eps

where C_i in [0, 1] is the local density (congestion) of agent i:

    C_i = 1/(P-1) * sum_{j != i} exp(-||X_i - X_j||^2 / sigma^2)
"""
import numpy as np


def mfo(fitness, dim, lb, ub, pop_size=30, iters=50,
        alpha=1.5, beta=0.5, gamma=0.1, sigma=None, seed=None, maximize=True):
    """Run MFO.

    fitness : callable, maps a 1-D array of length `dim` to a scalar.
    lb, ub  : scalars or arrays of length `dim` (search bounds).
    sigma   : crowding radius; default is 10% of the search-space diagonal.
    Returns (best_x, best_f, history) where history[t] is the best value
    after iteration t.
    """
    rng = np.random.default_rng(seed)
    lb = np.broadcast_to(np.asarray(lb, float), (dim,))
    ub = np.broadcast_to(np.asarray(ub, float), (dim,))
    span = ub - lb
    if sigma is None:
        sigma = 0.1 * np.linalg.norm(span)
    sign = 1.0 if maximize else -1.0

    X = lb + rng.random((pop_size, dim)) * span
    F = np.array([sign * fitness(x) for x in X])
    b = int(np.argmax(F))
    best_x, best_f = X[b].copy(), F[b]
    history = []

    for _ in range(iters):
        group = X.mean(axis=0)
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        C = (np.exp(-d2 / sigma ** 2).sum(1) - 1.0) / (pop_size - 1)

        r1, r2, r3 = (rng.random((pop_size, 1)) for _ in range(3))
        eps = rng.standard_normal((pop_size, dim)) * span
        move = alpha * r1 * (best_x - X) + beta * r2 * (group - X)
        X = X + (1 - C)[:, None] * move + C[:, None] * gamma * r3 * eps
        X = np.clip(X, lb, ub)

        F = np.array([sign * fitness(x) for x in X])
        b = int(np.argmax(F))
        if F[b] > best_f:
            best_x, best_f = X[b].copy(), F[b]
        history.append(sign * best_f)

    return best_x, sign * best_f, np.array(history)
