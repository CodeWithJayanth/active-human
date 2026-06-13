"""Demonstration-library generation for cache build.

sample_candidates() and build_demo_library() generate the synthetic demo set
that the ProMP is fit to. Invoked by build_cache.py. Build-time only; not used
at prediction time.
"""

import numpy as np
import time

from arm_predictor.config import CONFIG, JOINT_LIMITS
from arm_predictor.kinematics import lhs_samples, min_jerk_trajectory
from arm_predictor.optimizer import optimize_trajectory
from arm_predictor.costs import compute_ref_costs



def sample_candidates(K, seed, q0):
    rng = np.random.default_rng(seed)
    r = CONFIG["style_ranges"]
    max_disp = np.array(CONFIG["max_displacement"])
    min_disp = CONFIG["min_displacement"]

    ew  = lhs_samples(K, *r["effort_weight"], rng)
    sw  = lhs_samples(K, *r["speed_weight"], rng)
    lw  = lhs_samples(K, *r["limit_weight"], rng)
    sr  = lhs_samples(K, *r["shoulder_ratio"], rng)
    gw  = lhs_samples(K, *r["goal_weight"], rng)
    cw  = lhs_samples(K, *r["comfort_weight"], rng)
    dur = lhs_samples(K, *r["duration"], rng)
    cid = rng.integers(0, 3, size=K)

    candidates = []
    for i in range(K):
        while True:
            disp = rng.uniform(-max_disp, max_disp)
            if np.all(np.abs(disp) >= min_disp):
                break
        g = np.clip(q0 + disp, JOINT_LIMITS[:, 0] + 0.1, JOINT_LIMITS[:, 1] - 0.1)

        candidates.append({
            "q_goal": g,
            "effort_weight":  float(ew[i]),
            "speed_weight":   float(sw[i]),
            "limit_weight":   float(lw[i]),
            "shoulder_ratio": float(sr[i]),
            "goal_weight":    float(gw[i]),
            "comfort_weight": float(cw[i]),
            "duration":       float(dur[i]),
            "comfort_id":     int(cid[i]),
        })
    return candidates


def build_demo_library(q0, ref, candidates):
    N_phase = CONFIG["phase_points"]
    N_solve = CONFIG["solve_steps"]

    ref_disp = np.array([0.5, 0.4, 0.6, 0.2])
    _, qd_ref, _ = min_jerk_trajectory(q0, q0 + ref_disp, 2.0, N_solve)
    max_vel = 4.0 * np.max(np.abs(qd_ref))

    demos = []
    rej = {"goal": 0, "vel": 0, "fail": 0}
    t0 = time.time()

    for i, cand in enumerate(candidates):
        qf_i = cand["q_goal"]
        ref_i = compute_ref_costs(q0, qf_i, T=cand["duration"])
        try:
            q, qd, cost = optimize_trajectory(q0, qf_i, ref_i, cand)
        except Exception:
            rej["fail"] += 1
            continue

        goal_tol = max(0.05, 0.08 * np.linalg.norm(qf_i - q0))
        if np.linalg.norm(q[-1] - qf_i) > goal_tol:
            rej["goal"] += 1
            continue
        if np.max(np.abs(qd)) > max_vel:
            rej["vel"] += 1
            continue

        M = len(q) - 1
        phase_old = np.linspace(0, 1, M + 1)
        phase_new = np.linspace(0, 1, N_phase)
        q_ph = np.zeros((N_phase, 4))
        for j in range(4):
            q_ph[:, j] = np.interp(phase_new, phase_old, q[:, j])

        q_ph_delta = q_ph - q0

        demos.append({"q_phase": q_ph_delta, "T": cand["duration"], "q_goal": qf_i, "q_start": q0.copy()})

        if (i + 1) % 20 == 0:
            print(f"    [{i+1}/{len(candidates)}] valid={len(demos)} ({time.time()-t0:.0f}s)")

    print(f"    Done: {len(demos)} valid, rejected: {rej} ({time.time()-t0:.0f}s)")
    return demos
