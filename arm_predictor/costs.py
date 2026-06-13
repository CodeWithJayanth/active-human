"""Reference-cost computation for trajectory optimization.

compute_ref_costs() produces the reference-cost denominators used by the
L-BFGS-B trajectory optimizer during cache build. Build-time only; not used
at prediction time.
"""

import numpy as np

from arm_predictor.config import CONFIG, JOINT_LIMITS, COMFORT_POSES, SHOULDER, ELBOW
from arm_predictor.kinematics import min_jerk_trajectory



def compute_ref_costs(q0, qf, T=2.0):
    N = CONFIG["solve_steps"]
    q, qd, qdd = min_jerk_trajectory(q0, qf, T, N)
    eps = 0.05
    qc = COMFORT_POSES[0]

    sums = {"goal": 0, "speed": 0, "effort": 0, "comfort": 0,
            "shoulder": 0, "elbow": 0, "jlimit": 0}

    for t in range(N):
        qt, qdt, u = q[t+1], qd[t+1], qdd[t]
        sums["goal"]     += np.sum((qt - qf)**2)
        sums["speed"]    += np.sum(qdt**2)
        sums["effort"]   += np.sum(u**2)
        sums["comfort"]  += np.sum((qt - qc)**2)
        sums["shoulder"] += np.sum(u[SHOULDER]**2)
        sums["elbow"]    += np.sum(u[ELBOW]**2)
        d_lo = np.maximum(qt - JOINT_LIMITS[:, 0] + eps, 1e-6)
        d_hi = np.maximum(JOINT_LIMITS[:, 1] - qt + eps, 1e-6)
        sums["jlimit"]   += np.sum(1/d_lo**2 + 1/d_hi**2)

    return {k: max(v / N, 1e-12) for k, v in sums.items()}
