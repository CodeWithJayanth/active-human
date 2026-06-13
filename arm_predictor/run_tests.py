"""17-case ArmPredictor accuracy harness.

For each case, prints one compact block: case setup (start/goal in rad,
duration), per-joint position MAE (deg), and per-joint velocity MAE
(deg/s) — both at the 0.5 s and 1.0 s horizons. No baselines, no
aggregate row.

Usage:
    python -m arm_predictor.run_tests --group all
    python -m arm_predictor.run_tests --group D
"""

import argparse
import contextlib
import io
import math
import os
import sys

import numpy as np

from arm_predictor.config import CONFIG, JOINT_LIMITS, CACHE_FILE
from arm_predictor.kinematics import (min_jerk, finite_diff_vel,
                                       pd_controller_trajectory,
                                       sequential_joint_trajectory,
                                       add_obs_noise)
from arm_predictor.evaluation import run_prediction
from arm_predictor.cache_utils import load_cache


RAD2DEG = 180.0 / math.pi


def _gen_minjerk(q0, qf, T, dt):
    M = int(math.floor(T / dt)) + 1
    q = np.zeros((M, 4))
    for i in range(M):
        s = min(i * dt / T, 1.0)
        q[i] = q0 + (qf - q0) * min_jerk(s)
    return q, finite_diff_vel(q, dt)


def _gen_pd(q0, qf, T, dt, Kp, Kd):
    M = int(math.floor(T / dt)) + 1
    q, qd = pd_controller_trajectory(q0, qf, T, dt, Kp=Kp, Kd=Kd)
    if len(q) > M:
        q, qd = q[:M], qd[:M]
    elif len(q) < M:
        pad = M - len(q)
        q = np.vstack([q, np.tile(q[-1], (pad, 1))])
        qd = np.vstack([qd, np.tile(qd[-1], (pad, 1))])
    return q, qd


def _gen_seq_minjerk(q0, qf, T, dt, lead_joints):
    
    q = sequential_joint_trajectory(np.asarray(q0), np.asarray(qf), T, dt,
                                     lead_joints=lead_joints, lag_fraction=0.3)
    qd = finite_diff_vel(q, dt)
    return q, qd


def _gen_minjerk_with_dwell(q0, qf, T_motion, T_hold, dt):
    
    q_motion, qd_motion = _gen_minjerk(q0, qf, T_motion, dt)
    n_hold = int(math.floor(T_hold / dt))
    q_hold = np.tile(q_motion[-1], (n_hold, 1))
    qd_hold = np.zeros((n_hold, 4))
    q = np.vstack([q_motion, q_hold])
    qd = np.vstack([qd_motion, qd_hold])
    return q, qd


def _gen_minjerk_with_locked_joint(q_start, qf, T, dt, locked_joint_idx):
    
    q, qdot = _gen_minjerk(q_start, qf, T, dt)
    q[:, locked_joint_idx] = q_start[locked_joint_idx]
    qdot[:, locked_joint_idx] = 0.0
    return q, qdot


def _gen_reversal(q_start, q_mid, q_end, T_forward, T_back, dt):
    
    q1, qd1 = _gen_minjerk(q_start, q_mid, T_forward, dt)
    q2, qd2 = _gen_minjerk(q_mid, q_end, T_back, dt)
    q    = np.concatenate([q1,  q2[1:]],  axis=0)
    qdot = np.concatenate([qd1, qd2[1:]], axis=0)
    return q, qdot


def _gen_strict_sequential(q_start, q_mid, q_end, T_first, T_pause, T_second, dt,
                           first_joints, second_joints):
    
    n_first  = int(round(T_first  / dt))
    n_pause  = int(round(T_pause  / dt))
    n_second = int(round(T_second / dt))
    N = n_first + n_pause + n_second + 1
    n_joints = len(q_start)

    q    = np.zeros((N, n_joints))
    qdot = np.zeros((N, n_joints))

    q1, qd1 = _gen_minjerk(q_start, q_mid, T_first, dt)
    q[:n_first + 1, :] = q1
    qdot[:n_first + 1, :] = qd1
    non_first = [j for j in range(n_joints) if j not in first_joints]
    for j in non_first:
        q[:n_first + 1, j] = q_start[j]
        qdot[:n_first + 1, j] = 0.0

    pose_after_phase1 = q[n_first].copy()
    for k in range(n_pause):
        idx = n_first + 1 + k
        if idx < N:
            q[idx, :] = pose_after_phase1
            qdot[idx, :] = 0.0

    q_phase3_start = pose_after_phase1.copy()
    q_phase3_end = pose_after_phase1.copy()
    for j in second_joints:
        q_phase3_end[j] = q_end[j]
    q3, qd3 = _gen_minjerk(q_phase3_start, q_phase3_end, T_second, dt)
    start_idx = n_first + n_pause + 1
    for k in range(len(q3)):
        idx = start_idx + k - 1
        if 0 <= idx < N:
            q[idx, :] = q3[k]
            qdot[idx, :] = qd3[k]

    return q, qdot


def _gen_stationary_with_twitch(q_start, qf, T, dt, twitch_joint_idx,
                                 twitch_start_t, twitch_duration, twitch_amplitude):
    
    q, qdot = _gen_minjerk(q_start, qf, T, dt)
    q[:, twitch_joint_idx] = q_start[twitch_joint_idx]
    qdot[:, twitch_joint_idx] = 0.0

    N = q.shape[0]
    t_start_idx = int(round(twitch_start_t / dt))
    t_end_idx   = int(round((twitch_start_t + twitch_duration) / dt))
    t_end_idx   = min(t_end_idx, N - 1)

    for n in range(t_start_idx, t_end_idx + 1):
        phase = (n - t_start_idx) / max(t_end_idx - t_start_idx, 1)
        twitch_pos = twitch_amplitude * np.sin(np.pi * phase)
        twitch_vel = twitch_amplitude * (np.pi / twitch_duration) * np.cos(np.pi * phase)
        q[n, twitch_joint_idx] = q_start[twitch_joint_idx] + twitch_pos
        qdot[n, twitch_joint_idx] = twitch_vel

    return q, qdot


def _clip(q):
    return np.clip(np.array(q, dtype=float),
                   JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


def build_cases(dt):
    
    cases = []

    A_specs = [
        ("A1", 3.0, [ 0.0,  0.2,  0.0,  0.3], [ 0.6,  0.8,  0.8,  1.0]),
        ("A2", 4.5, [-0.4,  0.1,  0.4,  0.5], [ 0.5,  0.9,  1.1,  1.2]),
        ("A3", 5.5, [ 0.3,  0.5,  0.9,  0.7], [ 1.0,  1.4,  1.3,  1.8]),
        ("A4", 7.0, [ 1.0,  1.4,  1.3,  1.8], [ 0.3,  0.5,  0.9,  0.7]),
    ]
    for cid, T, q0, qf in A_specs:
        q_t, qd_t = _gen_minjerk(_clip(q0), _clip(qf), T, dt)
        cases.append({"id": cid, "group": "A",
                      "label": f"MinJerk reach, T={T}s",
                      "q_true": q_t, "qdot_true": qd_t, "T": T})

    q_b1, qd_b1 = _gen_seq_minjerk(_clip([0.0, 0.3, 0.5, 0.5]),
                                    _clip([0.7, 0.9, 1.0, 1.4]),
                                    5.0, dt, lead_joints=[0, 1, 2])
    cases.append({"id": "B1", "group": "B",
                  "label": "Sequential shoulder-lead, T=5.0s",
                  "q_true": q_b1, "qdot_true": qd_b1, "T": 5.0})

    q_b2, qd_b2 = _gen_seq_minjerk(_clip([0.2, 0.4, 0.3, 0.4]),
                                    _clip([0.8, 0.6, 0.8, 1.6]),
                                    5.0, dt, lead_joints=[3])
    cases.append({"id": "B2", "group": "B",
                  "label": "Sequential elbow-lead, T=5.0s",
                  "q_true": q_b2, "qdot_true": qd_b2, "T": 5.0})

    q_b3, qd_b3 = _gen_pd(_clip([-0.2, 0.2, 0.2, 0.6]),
                           _clip([ 0.4, 0.7, 0.9, 1.3]),
                           3.5, dt, Kp=6.0, Kd=3.5)
    cases.append({"id": "B3", "group": "B",
                  "label": "PD-Medium (Kp=6) reach, T=3.5s",
                  "q_true": q_b3, "qdot_true": qd_b3, "T": 3.5})

    q_b4, qd_b4 = _gen_pd(_clip([0.1, 0.5, 0.6, 0.5]),
                           _clip([0.6, 1.0, 1.1, 1.2]),
                           5.0, dt, Kp=12.0, Kd=5.5)
    cases.append({"id": "B4", "group": "B",
                  "label": "PD-Stiff (Kp=12) reach, T=5.0s",
                  "q_true": q_b4, "qdot_true": qd_b4, "T": 5.0})

    q_c1, qd_c1 = _gen_minjerk_with_dwell(_clip([0.0, 0.3, 0.4, 0.4]),
                                           _clip([0.5, 0.7, 0.9, 1.1]),
                                           T_motion=3.0, T_hold=2.0, dt=dt)
    cases.append({"id": "C1", "group": "C",
                  "label": "Dwell: 3.0s reach + 2.0s hold (T_total=5.0s)",
                  "q_true": q_c1, "qdot_true": qd_c1, "T": 5.0})

    q_c2, qd_c2 = _gen_minjerk_with_dwell(_clip([-0.3, 0.2, 0.6, 0.5]),
                                           _clip([ 0.6, 0.9, 1.0, 1.5]),
                                           T_motion=4.0, T_hold=1.0, dt=dt)
    cases.append({"id": "C2", "group": "C",
                  "label": "Dwell: 4.0s reach + 1.0s hold (T_total=5.0s)",
                  "q_true": q_c2, "qdot_true": qd_c2, "T": 5.0})

    q_start_d1 = np.array([0.0, 0.3, 0.2, 0.5])
    q_goal_d1  = np.array([0.5, 0.8, 0.6, 0.5])
    T_d1 = 5.0
    q_true_d1, qdot_true_d1 = _gen_minjerk_with_locked_joint(
        q_start_d1, q_goal_d1, T_d1, dt, locked_joint_idx=3)
    assert (q_true_d1 >= JOINT_LIMITS[:, 0]).all() and \
           (q_true_d1 <= JOINT_LIMITS[:, 1]).all(), \
           "D1 trajectory violates joint limits"
    cases.append({
        "id": "D1", "group": "D",
        "label": "Stationary elbow, shoulders move",
        "q_start": q_start_d1, "q_goal": q_goal_d1, "T": T_d1,
        "q_true": q_true_d1, "qdot_true": qdot_true_d1,
    })

    q_start_d2 = np.array([0.2, 0.3, 0.2, 0.5])
    q_goal_d2  = np.array([0.2, 0.8, 0.6, 1.0])
    T_d2 = 5.0
    q_true_d2, qdot_true_d2 = _gen_minjerk_with_locked_joint(
        q_start_d2, q_goal_d2, T_d2, dt, locked_joint_idx=0)
    assert (q_true_d2 >= JOINT_LIMITS[:, 0]).all() and \
           (q_true_d2 <= JOINT_LIMITS[:, 1]).all(), \
           "D2 trajectory violates joint limits"
    cases.append({
        "id": "D2", "group": "D",
        "label": "Stationary sh_yaw, others move",
        "q_start": q_start_d2, "q_goal": q_goal_d2, "T": T_d2,
        "q_true": q_true_d2, "qdot_true": qdot_true_d2,
    })

    q_start_d3 = np.array([0.1, 0.4, 0.3, 0.6])
    q_goal_d3  = np.array([0.1, 0.9, 0.3, 1.2])
    T_d3 = 5.0
    q_true_d3, qdot_true_d3 = _gen_minjerk(q_start_d3, q_goal_d3, T_d3, dt)
    q_true_d3[:, 0] = q_start_d3[0]; qdot_true_d3[:, 0] = 0.0
    q_true_d3[:, 2] = q_start_d3[2]; qdot_true_d3[:, 2] = 0.0
    assert (q_true_d3 >= JOINT_LIMITS[:, 0]).all() and \
           (q_true_d3 <= JOINT_LIMITS[:, 1]).all(), \
           "D3 trajectory violates joint limits"
    cases.append({
        "id": "D3", "group": "D",
        "label": "Two stationary joints (sh_yaw, sh_roll)",
        "q_start": q_start_d3, "q_goal": q_goal_d3, "T": T_d3,
        "q_true": q_true_d3, "qdot_true": qdot_true_d3,
    })

    q_start_d4 = np.array([0.0, 0.3, 0.2, 0.4])
    q_mid_d4   = np.array([0.6, 0.8, 0.7, 1.1])
    q_end_d4   = np.array([0.1, 0.4, 0.3, 0.5])
    T_d4_total = 5.0
    q_true_d4, qdot_true_d4 = _gen_reversal(
        q_start_d4, q_mid_d4, q_end_d4,
        T_forward=2.0, T_back=3.0, dt=dt)
    assert (q_true_d4 >= JOINT_LIMITS[:, 0]).all() and \
           (q_true_d4 <= JOINT_LIMITS[:, 1]).all(), \
           "D4 trajectory violates joint limits"
    cases.append({
        "id": "D4", "group": "D",
        "label": "Reversal: reach 2s + retract 3s (T_total=5.0s)",
        "q_start": q_start_d4, "q_goal": q_end_d4, "T": T_d4_total,
        "q_true": q_true_d4, "qdot_true": qdot_true_d4,
    })

    q_start_d5 = np.array([-0.2, 0.2, 0.1, 0.3])
    q_goal_d5  = np.array([ 0.4, 0.9, 0.7, 1.3])
    T_d5 = 2.5
    q_true_d5, qdot_true_d5 = _gen_minjerk(q_start_d5, q_goal_d5, T_d5, dt)
    assert (q_true_d5 >= JOINT_LIMITS[:, 0]).all() and \
           (q_true_d5 <= JOINT_LIMITS[:, 1]).all(), \
           "D5 trajectory violates joint limits"
    cases.append({
        "id": "D5", "group": "D",
        "label": "Fast min-jerk, T=2.5s (OOD short)",
        "q_start": q_start_d5, "q_goal": q_goal_d5, "T": T_d5,
        "q_true": q_true_d5, "qdot_true": qdot_true_d5,
    })

    q_start_d6 = np.array([0.0, 0.3, 0.2, 0.5])
    q_mid_d6   = np.array([0.5, 0.8, 0.6, 0.5])
    q_end_d6   = np.array([0.5, 0.8, 0.6, 1.2])
    T_d6_total = 5.0
    q_true_d6, qdot_true_d6 = _gen_strict_sequential(
        q_start_d6, q_mid_d6, q_end_d6,
        T_first=1.5, T_pause=0.5, T_second=3.0, dt=dt,
        first_joints=[0, 1, 2], second_joints=[3])
    assert (q_true_d6 >= JOINT_LIMITS[:, 0]).all() and \
           (q_true_d6 <= JOINT_LIMITS[:, 1]).all(), \
           "D6 trajectory violates joint limits"
    cases.append({
        "id": "D6", "group": "D",
        "label": "Strict sequential: shoulders→pause→elbow",
        "q_start": q_start_d6, "q_goal": q_end_d6, "T": T_d6_total,
        "q_true": q_true_d6, "qdot_true": qdot_true_d6,
    })

    q_start_d7 = np.array([0.0, 0.3, 0.2, 0.5])
    q_goal_d7  = np.array([0.5, 0.8, 0.6, 0.5])
    T_d7 = 5.0
    q_true_d7, qdot_true_d7 = _gen_stationary_with_twitch(
        q_start_d7, q_goal_d7, T_d7, dt,
        twitch_joint_idx=3, twitch_start_t=2.0,
        twitch_duration=0.4, twitch_amplitude=0.4)
    assert (q_true_d7 >= JOINT_LIMITS[:, 0]).all() and \
           (q_true_d7 <= JOINT_LIMITS[:, 1]).all(), \
           "D7 trajectory violates joint limits"
    cases.append({
        "id": "D7", "group": "D",
        "label": "Stationary elbow + 0.4rad twitch at t=2.0s",
        "q_start": q_start_d7, "q_goal": q_goal_d7, "T": T_d7,
        "q_true": q_true_d7, "qdot_true": qdot_true_d7,
    })

    return cases


def _print_promp_report(case, result, dt):
    """Per-case report: prints setup (start/goal in rad, duration) plus
    per-joint position MAE (deg) and per-joint velocity MAE (deg/s) at
    0.5s and 1.0s. No baselines."""
    JOINTS = ["yaw", "pitch", "roll", "elbow"]
    HORIZONS = [0.5, 1.0]
    q_true = case["q_true"]
    qdot_true = case["qdot_true"]
    T_steps = len(q_true)

    pos = {}
    for h in HORIZONS:
        h_key = f"{h:.2f}s"
        h_fr = max(1, int(round(h / dt)))
        errs = []
        for n, entry in result["raw_preds"].items():
            if h_key in entry and (n + h_fr) < T_steps:
                errs.append(np.abs(np.asarray(entry[h_key]) - q_true[n + h_fr]))
        pos[h] = (np.mean(errs, axis=0) * RAD2DEG) if errs else np.full(4, float("nan"))

    vel = {}
    for h in HORIZONS:
        h_key = f"{h:.2f}s"
        h_fr = max(1, int(round(h / dt)))
        errs = []
        for n, entry in result["raw_vel_preds"].items():
            if h_key in entry and (n + h_fr) < len(qdot_true):
                errs.append(np.abs(np.asarray(entry[h_key]) - qdot_true[n + h_fr]))
        vel[h] = (np.mean(errs, axis=0) * RAD2DEG) if errs else np.full(4, float("nan"))

    q_start = case["q_start"] if "q_start" in case else q_true[0]
    q_goal  = case["q_goal"]  if "q_goal"  in case else q_true[-1]
    p05, p10 = pos[0.5], pos[1.0]
    v05, v10 = vel[0.5], vel[1.0]
    print(f"{case['id']:<4} {case.get('label','')}")
    print(f"     duration: {case['T']:.1f} s")
    print(f"     start (rad):  yaw {q_start[0]:7.3f}  pitch {q_start[1]:7.3f}  roll {q_start[2]:7.3f}  elbow {q_start[3]:7.3f}")
    print(f"     goal  (rad):  yaw {q_goal[0]:7.3f}  pitch {q_goal[1]:7.3f}  roll {q_goal[2]:7.3f}  elbow {q_goal[3]:7.3f}")
    print(f"     pos@0.5s (deg)   yaw {p05[0]:5.1f}  pitch {p05[1]:5.1f}  roll {p05[2]:5.1f}  elbow {p05[3]:5.1f}")
    print(f"     pos@1.0s (deg)   yaw {p10[0]:5.1f}  pitch {p10[1]:5.1f}  roll {p10[2]:5.1f}  elbow {p10[3]:5.1f}")
    print(f"     vel@0.5s (deg/s) yaw {v05[0]:5.1f}  pitch {v05[1]:5.1f}  roll {v05[2]:5.1f}  elbow {v05[3]:5.1f}")
    print(f"     vel@1.0s (deg/s) yaw {v10[0]:5.1f}  pitch {v10[1]:5.1f}  roll {v10[2]:5.1f}  elbow {v10[3]:5.1f}")
    print()


def parse_args():
    p = argparse.ArgumentParser(
        description="17-case ProMP predictor accuracy harness")
    p.add_argument("--horizon", type=float, required=False, default=0.5,
                   choices=[0.5, 1.0],
                   help="Display flag (retained for compatibility; "
                        "report always shows both 0.5s and 1.0s).")
    p.add_argument("--group", default="all",
                   choices=["A", "B", "C", "D", "all"],
                   help="Which group to run (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    dt = CONFIG["dt"]
    sigma = CONFIG["obs_noise"]

    if not os.path.exists(CACHE_FILE):
        print(f"  ERROR: cache not found at {CACHE_FILE}")
        print("  Run: python build_cache.py")
        sys.exit(1)
    cache = load_cache(CACHE_FILE, CONFIG)
    promp = cache["promp"]

    cases = build_cases(dt)
    if args.group != "all":
        cases = [c for c in cases if c["group"] == args.group]

    print("=" * 80)
    print(f"  ProMP accuracy· group={args.group} · "
          f"{len(cases)} cases · horizons 0.5s & 1.0s · σ={sigma} · dt={dt}")
    print("=" * 80)

    for case in cases:
        q_true = case["q_true"]; qdot_true = case["qdot_true"]; T = case["T"]
        q_noisy = add_obs_noise(q_true, sigma)
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_prediction(promp, q_true, qdot_true, dt, T,
                                     q_noisy=q_noisy, sigma_override=sigma,
                                     enable_dwell=False)
        _print_promp_report(case, result, dt)

   
if __name__ == "__main__":
    main()
