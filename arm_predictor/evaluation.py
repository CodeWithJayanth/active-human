"""Run the predictor across a full trajectory and collect its predictions.

run_prediction() feeds an arm trajectory to the OnlinePredictor one frame at
a time, and at every frame records what the predictor forecasts for each
future time horizon (0.5s, 1.0s, etc.). The test harness (run_tests.py) calls
this and compares the forecasts against what actually happened, to measure
accuracy. This file is for evaluation only — it is not needed to run the
predictor in a real application.
"""

import numpy as np
import time as _time

from arm_predictor.config import CONFIG
from arm_predictor.predictor import OnlinePredictor
from arm_predictor.kinematics import add_obs_noise



def run_prediction(promp, q_true, qdot_true, dt, T_true,
                   q_noisy=None, sigma_override=None, oracle_T=None,
                   enable_dwell=False):
    
    sigma = sigma_override if sigma_override is not None else CONFIG["obs_noise"]
    if q_noisy is None:
        q_noisy = add_obs_noise(q_true, sigma)
    horizons = CONFIG["lookahead_horizons"]

    T_steps = len(q_true)
    win_max = CONFIG["condition_window_max"]
    q_start = q_true[0]
    engine = OnlinePredictor(promp, q_start=q_start, enable_dwell=enable_dwell,
                             sigma=sigma_override)
    use_oracle = oracle_T is not None
    if use_oracle:
        engine.duration_candidates = np.array([oracle_T])
        engine.log_duration_belief = np.array([0.0])

    N = T_steps - 1
    preds = np.zeros((N, 4))
    vel_preds = np.zeros((N, 4))
    stds = np.zeros((N, 4))
    T_hist = np.zeros(N)
    preds_05 = np.zeros((N, 4))
    gt_05 = np.zeros((N, 4))
    valid_05 = np.full(N, False)

    horizon_frames = [max(1, int(round(h / dt))) for h in horizons]
    lookahead = {f"{h:.2f}s": {"promp": [], "linear": [], "persist": [],
                                "const_accel": [],
                                "promp_early": [], "promp_mid": [], "promp_late": [],
                                "promp_vel": [], "vel_hold": [], "vel_ca": []}
                 for h in horizons}

    raw_preds = {}
    raw_vel_preds = {}

    _step_times = []

    for n in range(N):
        _t0 = _time.perf_counter()
        t_n = n * dt
        start = max(0, n - win_max + 1)
        window = [(i * dt, q_noisy[i]) for i in range(start, n + 1)]

        out = engine.step(t_n, window, dt)
        preds[n] = out["q_pred"]
        stds[n] = out["q_std"]
        T_hist[n] = out["T_est"]

        T_best = out["T_est"]
        s_next_vel = min((t_n + dt) / T_best, 1.0) if T_best > 0 else 1.0
        if T_best > 0:
            vel_preds[n], _ = promp.predict_vel_at(s_next_vel, T_best, out["mu_c"])
        else:
            vel_preds[n] = 0.0

        top_k_info = out.get("top_k_info")

        def _horizon_pos(t_n_local, h_sec_local):
            
            if top_k_info is not None:
                q = np.zeros(4)
                for Tk, mu_c_k, Sig_c_k, pk in top_k_info:
                    s_k = min((t_n_local + h_sec_local) / Tk, 1.0) if Tk > 0 else 1.0
                    d_k, _ = promp.predict_at(s_k, mu_c_k, Sig_c_k)
                    q += pk * (q_start + d_k)
                return q
            s_fut = min((t_n_local + h_sec_local) / T_best, 1.0)
            d, _ = promp.predict_at(s_fut, out["mu_c"], out["Sig_c"])
            return q_start + d

        def _horizon_vel(t_n_local, h_sec_local):
            
            ds = 0.005
            def _vel_single(Tk, mu_c_k, Sig_c_k):
                s_k = min((t_n_local + h_sec_local) / Tk, 1.0) if Tk > 0 else 1.0
                if s_k >= 1.0:
                    t_over = (t_n_local + h_sec_local) - Tk
                    s_bwd = 1.0 - ds
                    q_fwd, _ = promp.predict_at(1.0, mu_c_k, Sig_c_k)
                    q_bwd, _ = promp.predict_at(s_bwd, mu_c_k, Sig_c_k)
                    v_term = (q_fwd - q_bwd) / (ds * Tk)
                    return v_term * np.exp(-max(t_over, 0.0) / 0.4)
                s_fwd = min(s_k + ds, 1.0)
                s_bwd = max(s_k - ds, 0.0)
                q_fwd, _ = promp.predict_at(s_fwd, mu_c_k, Sig_c_k)
                q_bwd, _ = promp.predict_at(s_bwd, mu_c_k, Sig_c_k)
                gap = s_fwd - s_bwd
                if gap <= 0:
                    return np.zeros(4)
                v = (q_fwd - q_bwd) / (gap * Tk)
                return v

            if top_k_info is not None:
                v = np.zeros(4)
                for Tk, mu_c_k, Sig_c_k, pk in top_k_info:
                    v += pk * _vel_single(Tk, mu_c_k, Sig_c_k)
                return v
            return _vel_single(T_best, out["mu_c"], out["Sig_c"])

        h_plot = 0.5
        h_plot_frames = int(round(h_plot / dt))
        target_plot = n + h_plot_frames
        if target_plot < T_steps:
            preds_05[n] = _horizon_pos(t_n, h_plot)
            gt_05[n] = q_true[target_plot]
            valid_05[n] = True

        third = N // 3
        for h_sec, h_fr in zip(horizons, horizon_frames):
            target = n + h_fr
            if target >= T_steps:
                continue
            key = f"{h_sec:.2f}s"
            q_fut = _horizon_pos(t_n, h_sec)
            if n not in raw_preds:
                raw_preds[n] = {}
            raw_preds[n][key] = q_fut.copy()
            promp_err = np.mean(np.abs(q_fut - q_true[target]))
            lookahead[key]["promp"].append(promp_err)

            if n < third:
                lookahead[key]["promp_early"].append(promp_err)
            elif n < 2 * third:
                lookahead[key]["promp_mid"].append(promp_err)
            else:
                lookahead[key]["promp_late"].append(promp_err)

            q_persist = q_noisy[n]
            lookahead[key]["persist"].append(
                np.mean(np.abs(q_persist - q_true[target])))

            if n > 0:
                q_lin = q_noisy[n] + (q_noisy[n] - q_noisy[n-1]) * h_fr
                lin_err = np.mean(np.abs(q_lin - q_true[target]))
                lookahead[key]["linear"].append(lin_err)

            if n >= 2:
                _win = min(10, n + 1)
                _start = n - _win + 1
                _window = q_noisy[_start:n + 1]
                _t_local = np.arange(_win) * dt
                _t_end = _t_local[-1]
                _v_ca = np.zeros(4); _a_ca = np.zeros(4)
                for _j in range(4):
                    _c = np.polyfit(_t_local, _window[:, _j], 2)
                    _a_ca[_j] = 2 * _c[0]
                    _v_ca[_j] = 2 * _c[0] * _t_end + _c[1]
                q_ca = q_noisy[n] + _v_ca * h_sec + 0.5 * _a_ca * h_sec**2
                lookahead[key]["const_accel"].append(np.mean(np.abs(q_ca - q_true[target])))

            if target < len(qdot_true) and T_best > 0:
                vel_pred_h = _horizon_vel(t_n, h_sec)
                raw_vel_preds.setdefault(n, {})[key] = vel_pred_h.copy()
                lookahead[key]["promp_vel"].append(
                    np.mean(np.abs(vel_pred_h - qdot_true[target])))
                if n > 0:
                    v_hold = (q_noisy[n] - q_noisy[n-1]) / dt
                    lookahead[key]["vel_hold"].append(
                        np.mean(np.abs(v_hold - qdot_true[target])))

                    if n >= 2:
                        a_est = (q_noisy[n] - 2*q_noisy[n-1] + q_noisy[n-2]) / (dt * dt)
                        v_ca = v_hold + a_est * h_sec
                    else:
                        v_ca = v_hold
                    lookahead[key]["vel_ca"].append(
                        np.mean(np.abs(v_ca - qdot_true[target])))

        _step_times.append(_time.perf_counter() - _t0)

    if _step_times:
        avg_ms = np.mean(_step_times) * 1000
        max_ms = np.max(_step_times) * 1000
        print(f"  Runtime: {avg_ms:.2f} ms/step avg, {max_ms:.2f} ms/step max "
              f"({1000/avg_ms:.0f} Hz capable)")
    else:
        avg_ms = 0.0
        max_ms = 0.0

    return {
        "preds": preds, "vel_preds": vel_preds,
        "stds": stds, "T_hist": T_hist,
        "q_noisy": q_noisy,
        "lookahead": lookahead,
        "T_est_final": float(T_hist[-1]) if len(T_hist) else float("nan"),
        "preds_05": preds_05, "gt_05": gt_05, "valid_05": valid_05,
        "raw_preds": raw_preds,
        "raw_vel_preds": raw_vel_preds,
        "enable_dwell": enable_dwell,
        "dwell_active": bool(engine.dwell_active),
    }
