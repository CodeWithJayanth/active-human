import numpy as np

from arm_predictor.config import CONFIG




class ProMP:
    def __init__(self):
        B = CONFIG["num_basis"]
        N = CONFIG["phase_points"]
        self.num_basis = B
        self.num_joints = 4
        self.s_grid = np.linspace(0, 1, N)

        centers = np.linspace(0, 1, B)
        spacing = centers[1] - centers[0] if B > 1 else 0.5
        width = spacing * CONFIG["basis_width"]

        self.basis_matrix = np.zeros((N, B))
        for b in range(B):
            self.basis_matrix[:, b] = np.exp(-0.5 * ((self.s_grid - centers[b]) / width)**2)
        self.basis_matrix /= np.maximum(self.basis_matrix.sum(axis=1, keepdims=True), 1e-10)
        self.centers = centers
        self.width = width
        self.weight_mean = None
        self.weight_cov = None

    def _basis_at(self, s):
        phi = np.exp(-0.5 * ((s - self.centers) / self.width)**2)
        phi /= max(phi.sum(), 1e-10)
        return phi

    def _block_basis(self, Phi_single):
        M = Phi_single.shape[0]
        Phi_block = np.zeros((M * self.num_joints, self.num_basis * self.num_joints))
        for d in range(self.num_joints):
            Phi_block[d*M:(d+1)*M, d*self.num_basis:(d+1)*self.num_basis] = Phi_single
        return Phi_block

    def fit(self, demo_positions):
        K = len(demo_positions)
        Phi_block = self._block_basis(self.basis_matrix)
        W = np.zeros((K, self.num_basis * self.num_joints))
        for k in range(K):
            y = demo_positions[k].T.ravel()
            W[k] = np.linalg.lstsq(Phi_block, y, rcond=None)[0]
        self.weight_mean = np.mean(W, axis=0)
        if K > 1:
            diff = W - self.weight_mean
            self.weight_cov = (diff.T @ diff) / (K - 1)
        else:
            self.weight_cov = 0.01 * np.eye(self.num_basis * self.num_joints)
        self.weight_cov += 1e-6 * np.eye(self.num_basis * self.num_joints)

        
        nb = self.num_basis
        for i in range(self.num_joints):
            for j in range(self.num_joints):
                if i != j:
                    self.weight_cov[i*nb:(i+1)*nb, j*nb:(j+1)*nb] = 0.0

    def condition(self, s_obs, y_obs, sigma=None):
        if sigma is None:
            sigma = CONFIG["obs_noise"]
        s_obs = np.atleast_1d(s_obs)
        y_obs = np.atleast_2d(y_obs)  # (num_obs, 4)
        num_obs = len(s_obs)

        basis_at_obs = np.zeros((num_obs, self.num_basis))
        for m in range(num_obs):
            basis_at_obs[m] = self._basis_at(s_obs[m])

        # Block-diagonal layout — same ordering as _block_basis and fit()
        # obs_flat: [j0_obs0, j0_obs1, ..., j1_obs0, j1_obs1, ...]
        obs_matrix = self._block_basis(basis_at_obs)
        obs_flat = y_obs.T.ravel()

        obs_noise_cov = sigma**2 * np.eye(num_obs * self.num_joints)

        S = obs_matrix @ self.weight_cov @ obs_matrix.T + obs_noise_cov
        try:
            S_inv = np.linalg.solve(S, np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K_gain = self.weight_cov @ obs_matrix.T @ S_inv
        innovation = obs_flat - obs_matrix @ self.weight_mean

        mu_new = self.weight_mean + K_gain @ innovation
        Sigma_new = self.weight_cov - K_gain @ obs_matrix @ self.weight_cov
        Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
        return mu_new, Sigma_new

    def predict_vel_at(self, s, T, mu_w=None, Sigma_w=None):
        
        if mu_w is None: mu_w = self.weight_mean
        if Sigma_w is None: Sigma_w = self.weight_cov
        phi = np.exp(-0.5 * ((s - self.centers) / self.width)**2)
        phi_sum = max(phi.sum(), 1e-10)
        dphi_raw = -(s - self.centers) / self.width**2 * phi
        
        dphi_norm = (dphi_raw * phi_sum - phi * dphi_raw.sum()) / phi_sum**2
        qdot = np.zeros(self.num_joints)
        qdot_std = np.zeros(self.num_joints)
        nb = self.num_basis
        for d in range(self.num_joints):
            w_d = mu_w[d*nb:(d+1)*nb]
            qdot[d] = dphi_norm @ w_d / T
            S_d = Sigma_w[d*nb:(d+1)*nb, d*nb:(d+1)*nb]
            qdot_std[d] = np.sqrt(max(dphi_norm @ S_d @ dphi_norm, 0.0)) / T
        return qdot, qdot_std

    def predict_at(self, s, mu_w=None, Sigma_w=None):
        if mu_w is None: mu_w = self.weight_mean
        if Sigma_w is None: Sigma_w = self.weight_cov
        basis_vals = self._basis_at(s)
        q = np.zeros(self.num_joints)
        std = np.zeros(self.num_joints)
        for d in range(self.num_joints):
            w_d = mu_w[d*self.num_basis:(d+1)*self.num_basis]
            q[d] = basis_vals @ w_d
            S_d = Sigma_w[d*self.num_basis:(d+1)*self.num_basis,
                          d*self.num_basis:(d+1)*self.num_basis]
            std[d] = np.sqrt(max(basis_vals @ S_d @ basis_vals, 0))
        return q, std
