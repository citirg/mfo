# MFO — Mudik Flow Optimization

Minimal Python implementation of **Mudik Flow Optimization (MFO)**, a
human-behavior-inspired metaheuristic based on the collective movement of
travellers during the Indonesian *mudik* (homecoming) season, applied to
weighted sum rate (WSR) maximization in downlink multicarrier NOMA.

Each agent (1) moves toward the destination (global best), (2) follows the
group flow (population mean), and (3) avoids congestion: a density factor
`C_i` switches crowded agents from exploitation to exploration.

## Files

| File | Content |
|---|---|
| `mfo.py` | MFO optimizer (generic, works with any fitness function) |
| `noma.py` | Minimal MC-NOMA WSR model with SIC and a per-subcarrier user limit |
| `demo.py` | Example run for several values of `P_max` |

## Usage

```bash
pip install -r requirements.txt
python demo.py
```

Using MFO on your own problem:

```python
from mfo import mfo
best_x, best_f, history = mfo(my_fitness, dim=10, lb=0, ub=1,
                              pop_size=30, iters=50, maximize=True)
```

This is a minimal reference implementation; parameter defaults
(`alpha`, `beta`, `gamma`, `sigma`) are illustrative.
