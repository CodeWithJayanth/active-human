"""Trajectory generators and kinematics helpers.

Provides motion-profile generators (min-jerk, PD-controller, sequential-joint),
finite-difference velocity, Latin-hypercube sampling, and observation-noise
injection. Consumed by the test harness (run_tests.py) and the cache-build chain.
"""

import numpy as np
import math

from arm_predictor.config import JOINT_LIMITS



def min_jerk(s):
    return 10*s**3 - 15*s**4 + 6*s**5

def min_jerk_trajectory(q0, qf, T, N):
    tau = np.linspace(0, 1, N + 1)
    s = min_jerk(tau)
    ds = (30*tau**2 - 60*tau**3 + 30*tau**4) / T
    dds = (60*tau - 180*tau**2 + 120*tau**3) / T**2
    q = q0 + s[:, None] * (qf - q0)
    qd = ds[:, None] * (qf - q0)
    qdd = dds[:, None] * (qf - q0)
    return q, qd, qdd

def finite_diff_vel(q, dt):
    v = np.zeros_like(q)
    v[0] = (q[1] - q[0]) / dt
    v[-1] = (q[-1] - q[-2]) / dt
    v[1:-1] = (q[2:] - q[:-2]) / (2 * dt)
    return v

def lhs_samples(K, lo, hi, rng):
    edges = np.linspace(0, 1, K + 1)
    u = rng.random(K)
    pts = edges[:K] + u * (edges[1:] - edges[:K])
    rng.shuffle(pts)
    return lo + (hi - lo) * pts


def pd_controller_trajectory(q0, qf, T, dt, Kp=12.0, Kd=5.5):
    
    N = int(round(T / dt))
    q = np.zeros((N + 1, 4))
    qd = np.zeros((N + 1, 4))
    q[0] = q0.copy()
    for i in range(N):
        acc = Kp * (qf - q[i]) - Kd * qd[i]
        qd[i + 1] = qd[i] + acc * dt
        q[i + 1] = q[i] + qd[i + 1] * dt
        q[i + 1] = np.clip(q[i + 1], JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        qd[i + 1] = (q[i + 1] - q[i]) / dt
    return q, qd


def sequential_joint_trajectory(q0, qf, T, dt, lead_joints, lag_fraction=0.3):
    
    M = int(math.floor(T / dt)) + 1
    q = np.zeros((M, 4))
    for i in range(M):
        t = i * dt
        for j in range(4):
            if j in lead_joints:
                s = min(t / T, 1.0)
            else:
                t_shifted = max(0, t - lag_fraction * T)
                T_remaining = T * (1 - lag_fraction)
                s = min(t_shifted / T_remaining, 1.0) if T_remaining > 0 else 1.0
            q[i, j] = q0[j] + (qf[j] - q0[j]) * min_jerk(s)
    return q


def add_obs_noise(q_true, sigma, bias_std=0.0, seed=123):
   
    rng = np.random.default_rng(seed)
    q_noisy = q_true.copy()
    if bias_std > 0:
        bias = rng.normal(0, bias_std, size=q_true.shape[1])
        q_noisy = q_noisy + bias
    q_noisy = q_noisy + rng.normal(0, sigma, q_true.shape)
    return q_noisy
