"""The arm-motion predictor  this is the main entry point for using the model.

Create an OnlinePredictor with the loaded ProMP and the known starting pose,
then call step() repeatedly as new observations of the arm arrive. Each call
takes the recent observation window and returns the predicted next pose plus
the internal belief state (estimated motion duration and the conditioned
weight posterior) that downstream horizon predictions are built from.

This is the only file an application needs to call directly. It does not load
the cache itself — the caller loads promp_cache.pkl (via cache_utils) and
passes the resulting ProMP in.
"""

import numpy as np

from arm_predictor.config import CONFIG, JOINT_LIMITS



class OnlinePredictor:
    def __init__(self, promp, q_start, enable_dwell=False, sigma=None):
        self.promp = promp
        self.q_start = np.array(q_start)
        self.sigma = sigma if sigma is not None else CONFIG["obs_noise"]
        self.duration_candidates = np.linspace(CONFIG["T_min"], CONFIG["T_max"],
                                               CONFIG["num_T_hypotheses"])
        _mu_T  = CONFIG["duration_prior_mean"]
        _sig_T = CONFIG["duration_prior_std"]
        self.log_duration_belief = -0.5 * ((self.duration_candidates - _mu_T) / _sig_T) ** 2
        self.enable_dwell = enable_dwell
        self.dwell_active = False
        self.dwell_position = None
        self.dwell_T_locked = None
        self.dwell_mu_c = None
        self.dwell_Sig_c = None
        self._vel_below_count = 0
        _joint_ranges = JOINT_LIMITS[:, 1] - JOINT_LIMITS[:, 0]
        self._dwell_vel_thresh = CONFIG["dwell_vel_fraction"] * _joint_ranges
        self._dwell_dwell_sec = 0.4
        self._min_phase_for_dwell = 0.3
        self._dwell_min_elapsed = 0.5
        self._dwell_belief_threshold = 0.15
        self._weighted_cutoff_sec = 2.0
        self._step_count = 0
        self._last_mu_c = self.promp.weight_mean.copy()
        self._last_Sig_c = self.promp.weight_cov.copy()
        self._prev_q_obs = None
        self._posterior_per_Tk = None

    @property
    def T_est(self):
        return float(self.duration_candidates[np.argmax(self.log_duration_belief)])

    def _check_dwell(self, obs_window, dt):
        """Check if arm has stopped moving using smoothed velocity."""
        if len(obs_window) < 5:
            return False

        t_now = obs_window[-1][0]
        if t_now < self._dwell_min_elapsed:
            return False
        log_b = self.log_duration_belief
        probs = np.exp(log_b - np.max(log_b))
        probs /= probs.sum()
        if probs.max() < self._dwell_belief_threshold:
            return False

        _, q_first = obs_window[0]
        _, q_last = obs_window[-1]
        t_first = obs_window[0][0]
        t_last = obs_window[-1][0]
        dt_win = t_last - t_first
        if dt_win < 1e-6:
            return False

        smoothed_vel = np.abs(q_last - q_first) / dt_win

        if np.all(smoothed_vel < self._dwell_vel_thresh):
            self._vel_below_count += 1
        else:
            self._vel_below_count = 0

        T_current = self.T_est
        t_now = obs_window[-1][0]
        phase_est = t_now / T_current if T_current > 0 else 0

        dwell_frames_req = max(1, int(round(self._dwell_dwell_sec / dt)))
        return (self._vel_below_count >= dwell_frames_req
                and phase_est >= self._min_phase_for_dwell)

    def _condition_or_stash(self, s_list, y_list):
       
        if s_list:
            return self.promp.condition(np.array(s_list), np.array(y_list),
                                        sigma=self.sigma)
        return self._last_mu_c, self._last_Sig_c

    def step(self, t_now, obs_window, dt):
        if not obs_window:
            raise ValueError("OnlinePredictor.step requires non-empty obs_window")
        sigma = self.sigma
        _, q_now = obs_window[-1]
        delta_now = q_now - self.q_start

        if self._prev_q_obs is not None and dt > 0:
            qdot_obs = (q_now - self._prev_q_obs) / dt
        else:
            qdot_obs = None
        self._prev_q_obs = q_now.copy()
        vel_weight = CONFIG["velocity_likelihood_weight"]
        vel_sigma = CONFIG["velocity_noise_sigma"]

        if self.dwell_active:
            q_pred = self.dwell_position.copy()
            q_std = np.zeros(4)
            return {"q_pred": q_pred, "q_std": q_std, "T_est": self.dwell_T_locked,
                    "mu_c": self.dwell_mu_c, "Sig_c": self.dwell_Sig_c,
                    "top_k_info": None}

        if self._posterior_per_Tk is None:
            self._posterior_per_Tk = [
                {'mu': self.promp.weight_mean.copy(),
                 'Sig': self.promp.weight_cov.copy()}
                for _ in range(len(self.duration_candidates))
            ]

        nb = self.promp.num_basis
        nj = self.promp.num_joints
        obs_noise_cov = (sigma ** 2) * np.eye(nj)
        for k, Tk in enumerate(self.duration_candidates):
            sk = t_now / Tk
            if sk > 1.05:
                self.log_duration_belief[k] += -50
                continue
            sk = min(sk, 1.0)
            post = self._posterior_per_Tk[k]
            mu_k, std_k = self.promp.predict_at(sk, post['mu'], post['Sig'])
            var_k = std_k ** 2 + sigma ** 2
            diff = delta_now - mu_k
            ll_pos = -0.5 * np.sum(diff ** 2 / var_k + np.log(2 * np.pi * var_k))
            ll_total = ll_pos

            if qdot_obs is not None:
                mu_v_k, std_v_k = self.promp.predict_vel_at(
                    sk, Tk, mu_w=post['mu'], Sigma_w=post['Sig'])
                var_v_k = std_v_k ** 2 + vel_sigma ** 2
                diff_v = qdot_obs - mu_v_k
                ll_vel = -0.5 * np.sum(
                    diff_v ** 2 / var_v_k + np.log(2 * np.pi * var_v_k))
                ll_total += vel_weight * ll_vel

            self.log_duration_belief[k] += ll_total

            phi = self.promp._basis_at(sk)
            H = np.zeros((nj, nb * nj))
            for d in range(nj):
                H[d, d * nb:(d + 1) * nb] = phi
            mu_w = post['mu']
            Sig_w = post['Sig']
            S = H @ Sig_w @ H.T + obs_noise_cov
            try:
                S_inv = np.linalg.solve(S, np.eye(nj))
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)
            K = Sig_w @ H.T @ S_inv
            innov = delta_now - H @ mu_w
            mu_w_new = mu_w + K @ innov
            I_minus_KH = np.eye(nb * nj) - K @ H
            Sig_w_new = I_minus_KH @ Sig_w @ I_minus_KH.T + K @ obs_noise_cov @ K.T
            self._posterior_per_Tk[k] = {'mu': mu_w_new, 'Sig': Sig_w_new}

        self.log_duration_belief -= np.max(self.log_duration_belief)

        self._step_count += 1
        weighted_cutoff_frames = max(1, int(round(self._weighted_cutoff_sec / dt)))
        use_weighted = (self._step_count <= weighted_cutoff_frames)

        log_b = self.log_duration_belief.copy()
        probs = np.exp(log_b)
        probs /= probs.sum()

        map_idx = int(np.argmax(probs))
        T_best = self.duration_candidates[map_idx]

        q_pred_weighted = np.zeros(4)
        q_std_weighted  = np.zeros(4)
        mu_c_best  = None
        Sig_c_best = None
        top_k_info = None

        if use_weighted:
            top_k = 3
            top_indices = np.argsort(probs)[-top_k:]
            top_probs = probs[top_indices]
            top_probs /= top_probs.sum()
            T_best = self.duration_candidates[top_indices[-1]]
            second_moment = np.zeros(4)
            top_k_info = []

            for idx, (ti, pi) in enumerate(zip(top_indices, top_probs)):
                Tk = self.duration_candidates[ti]

                s_list, y_list = [], []
                for t_i, q_i in obs_window:
                    si = t_i / Tk
                    if si <= 1.0:
                        s_list.append(si)
                        y_list.append(q_i - self.q_start)

                mu_c, Sig_c = self._condition_or_stash(s_list, y_list)
                top_k_info.append((float(Tk), mu_c, Sig_c, float(pi)))

                if idx == len(top_indices) - 1:
                    mu_c_best  = mu_c
                    Sig_c_best = Sig_c
                    if s_list:
                        self._last_mu_c = mu_c.copy()
                        self._last_Sig_c = Sig_c.copy()

                s_next = min((t_now + dt) / Tk, 1.0)
                delta_pred, std_pred = self.promp.predict_at(s_next, mu_c, Sig_c)
                q_pred_i = self.q_start + delta_pred
                q_pred_weighted += pi * q_pred_i
                second_moment   += pi * (q_pred_i**2 + std_pred**2)

            var_mix = np.maximum(second_moment - q_pred_weighted**2, 0.0)
            q_std_weighted = np.sqrt(var_mix)
        else:
            Tk_map = T_best
            s_list, y_list = [], []
            for t_i, q_i in obs_window:
                si = t_i / Tk_map
                if si <= 1.0:
                    s_list.append(si)
                    y_list.append(q_i - self.q_start)

            mu_c_best, Sig_c_best = self._condition_or_stash(s_list, y_list)
            if s_list:
                self._last_mu_c = mu_c_best.copy()
                self._last_Sig_c = Sig_c_best.copy()

            s_next = min((t_now + dt) / Tk_map, 1.0)
            delta_pred, std_pred = self.promp.predict_at(s_next, mu_c_best, Sig_c_best)
            q_pred_weighted = self.q_start + delta_pred

            top_k = 3
            top_indices = np.argsort(probs)[-top_k:]
            top_probs_local = probs[top_indices]
            top_probs_local /= top_probs_local.sum()
            second_moment = np.zeros(4)
            q_pred_mix = np.zeros(4)
            for ti, pi in zip(top_indices, top_probs_local):
                Tk_i = self.duration_candidates[ti]
                s_next_i = min((t_now + dt) / Tk_i, 1.0)
                d_i, s_i = self.promp.predict_at(s_next_i, mu_c_best, Sig_c_best)
                q_i = self.q_start + d_i
                q_pred_mix    += pi * q_i
                second_moment += pi * (q_i**2 + s_i**2)
            var_mix = np.maximum(second_moment - q_pred_mix**2, 0.0)
            q_std_weighted = np.sqrt(var_mix)

        if self.enable_dwell and self._check_dwell(obs_window, dt):
            self.dwell_active = True
            self.dwell_position = q_now.copy()
            self.dwell_T_locked = T_best
            self.dwell_mu_c  = mu_c_best
            self.dwell_Sig_c = Sig_c_best
            return {"q_pred": q_now.copy(), "q_std": np.zeros(4), "T_est": T_best,
                    "mu_c": mu_c_best, "Sig_c": Sig_c_best, "top_k_info": None}

        return {"q_pred": q_pred_weighted, "q_std": q_std_weighted, "T_est": T_best,
                "mu_c": mu_c_best, "Sig_c": Sig_c_best, "top_k_info": top_k_info}
