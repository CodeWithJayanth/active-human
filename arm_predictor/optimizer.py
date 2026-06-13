"""L-BFGS-B trajectory optimizer.

optimize_trajectory() solves for a movement trajectory matching a target
start/goal and style, used to synthesize the demonstration library during
cache build. Build-time only; not used at prediction time.
"""

import numpy as np
from scipy.optimize import minimize

from arm_predictor.config import CONFIG, JOINT_LIMITS, COMFORT_POSES, SHOULDER, ELBOW
from arm_predictor.kinematics import min_jerk_trajectory



def optimize_trajectory(q0, qf, ref, style):
    N = CONFIG["solve_steps"]
    T = style["duration"]
    dt = T / N

    comfort_pose = COMFORT_POSES[style["comfort_id"]]
    eps = 0.05
    w_goal     = style["goal_weight"]    / ref["goal"]
    w_speed    = style["speed_weight"]   / ref["speed"]
    w_effort   = style["effort_weight"]  / ref["effort"]
    w_comfort  = style["comfort_weight"] / ref["comfort"]
    w_shoulder = style["shoulder_ratio"] / ref["shoulder"]
    w_elbow    = (1 - style["shoulder_ratio"]) / ref["elbow"]
    w_limit    = style["limit_weight"]   / ref["jlimit"]

    _, _, accel_init = min_jerk_trajectory(q0, qf, T, N)
    accel_flat0 = accel_init[:N].ravel()

    def rollout(u_flat):
        u = u_flat.reshape(N, 4)
        vel = np.zeros((N + 1, 4))
        vel[1:] = np.cumsum(u, axis=0) * dt
        pos = np.zeros((N + 1, 4))
        pos[0] = q0
        pos[1:] = q0 + np.cumsum(vel[:-1] * dt + 0.5 * u * dt**2, axis=0)
        return pos, vel, u

    def cost(u_flat):
        pos, vel, u = rollout(u_flat)
        pt, vt = pos[1:], vel[1:]
        J  = w_goal    * np.sum((pt - qf)**2)
        J += w_speed   * np.sum(vt**2)
        J += w_effort  * np.sum(u**2)
        J += w_comfort * np.sum((pt - comfort_pose)**2)
        J += w_shoulder * np.sum(u[:, SHOULDER]**2)
        J += w_elbow   * np.sum(u[:, ELBOW]**2)
        d_lo = np.maximum(pt - JOINT_LIMITS[:, 0] + eps, 1e-6)
        d_hi = np.maximum(JOINT_LIMITS[:, 1] - pt + eps, 1e-6)
        J += w_limit * np.sum(1/d_lo**2 + 1/d_hi**2)
        J *= dt
        disp_scale = max(np.sum((q0 - qf)**2), 0.04)
        J += 200 * np.sum((pos[-1] - qf)**2) / disp_scale
        J += 100 * np.sum(vel[-1]**2) / disp_scale
        return J

    result = minimize(cost, accel_flat0, method="L-BFGS-B",
                      options={"maxiter": 600, "ftol": 1e-10, "gtol": 1e-7})
    pos, vel, _ = rollout(result.x)
    return pos, vel, result.fun
