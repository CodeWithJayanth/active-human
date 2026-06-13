"""Central configuration for the arm predictor.

Defines CONFIG (all tunable parameters), JOINT_LIMITS (4-DOF biomechanical
bounds: sh_yaw, sh_pitch, sh_roll, elbow), and joint groupings. Imported by
every module in the package. The import-time assert validates that the
 q_start lies within JOINT_LIMITS.
"""

import numpy as np

CACHE_FILE = "promp_cache.pkl"


CONFIG = {
    "q_start": np.array([0.0, 0.3, 0.0, 0.52]),
    "dt": 0.02,

    "num_demos": 200,
    "demo_seed": 42,
    "solve_steps": 40,

    "max_displacement": np.array([1.30, 1.40, 1.40, 1.50]),
    "min_displacement": 0.1,

    "style_ranges": {
        "effort_weight":   (0.3, 3.0),
        "speed_weight":    (0.3, 3.0),
        "limit_weight":    (0.3, 3.0),
        "shoulder_ratio":  (0.0, 1.0),
        "goal_weight":     (0.5, 3.0),
        "comfort_weight":  (0.3, 3.0),
        "duration":        (1.0, 6.0),
    },

    "num_basis": 25,
    "basis_width": 0.5,
    "obs_noise": 0.004,
    "phase_points": 200,

    "T_min": 0.8,
    "T_max": 10.0,
    "num_T_hypotheses": 40,
    "condition_window_max": 10,

    "duration_prior_mean": 4.0,
    "duration_prior_std":  2.0,

    "velocity_likelihood_weight": 0.25,
    "velocity_noise_sigma": 0.2,

    "dwell_vel_fraction": 0.03,

    "lookahead_horizons": [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
}

JOINT_LIMITS = np.array([
    [-1.30899694, +2.26892803],
    [-0.17453293, +3.14159265],
    [-1.57079633, +1.57079633],
    [+0.08726646, +2.61799388],
])

COMFORT_POSES = np.array([
    [ 0.0,  0.2,  0.5,  0.0],
    [-0.2,  0.3,  0.8, -0.1],
    [ 0.1,  0.1,  0.3,  0.1],
])

SHOULDER = np.array([0, 1])
ELBOW = np.array([2, 3])


_q_start = np.array(CONFIG["q_start"])
assert np.all(_q_start >= JOINT_LIMITS[:, 0]), (
    f"q_start {CONFIG['q_start']} violates JOINT_LIMITS lower bounds "
    f"{JOINT_LIMITS[:, 0].tolist()}"
)
assert np.all(_q_start <= JOINT_LIMITS[:, 1]), (
    f"q_start {CONFIG['q_start']} violates JOINT_LIMITS upper bounds "
    f"{JOINT_LIMITS[:, 1].tolist()}"
)
