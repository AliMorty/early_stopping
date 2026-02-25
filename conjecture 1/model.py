import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy.linalg import norm


class OverparameterizedLogisticRegression:
    """
    Data generator and GD runner for overparameterized logistic regression
    following Assumption 1 from Wu et al. (2025).

    Data model:
        x ~ N(0, Sigma),  Pr(y|x) = 1 / (1 + exp(-y * x^T w*))

    Covariance Sigma = U diag(lambda_1, ..., lambda_d) U^T where U = I
    (we work directly in the eigenbasis).
    """

    def __init__(self, d=2000, n=1000, k=100, eta=None, seed=42):
        self.d = d
        self.n = n
        self.k = k
        self.seed = seed

        # Eigenvalues: lambda_i = i^{-2}
        self.eigenvalues = np.array([(i + 1) ** (-2) for i in range(d)])

        # True parameter in eigenbasis: first k components = 1, rest = 0
        self.w_star = np.zeros(d)
        self.w_star[:k] = 1.0

        # Step size: eta <= 1 / (C0 * (1 + tr(Sigma) + lambda_1 * ln(1/delta) / n))
        tr_sigma = np.sum(self.eigenvalues)
        lambda_1 = self.eigenvalues[0]
        delta = 0.01
        C0 = 2.0
        eta_upper = 1.0 / (C0 * (1 + tr_sigma + lambda_1 * np.log(1.0 / delta) / n))

        if eta is None:
            self.eta = eta_upper
        else:
            self.eta = min(eta, eta_upper)

        print(f'Parameters: d={d}, n={n}, k={k}')
        print(f'tr(Sigma) = {tr_sigma:.4f}')
        print(f'eta (used) = {self.eta:.6f}')

        # State
        self.X = None
        self.y = None
        self.w_current = None  # current GD iterate
        self.t_current = 0    # current GD step
        self.w_history = []   # list of (t, w_t)
        self.loss_history = [] # list of (t, loss)
        self.w_tilde = None

    def generate_data(self):
        """Generate n data points from the logistic model."""
        rng = np.random.RandomState(self.seed)
        Z = rng.randn(self.n, self.d)
        self.X = Z * np.sqrt(self.eigenvalues)[np.newaxis, :]

        # --- Alternative: library version using multivariate_normal ---
        # Works for any covariance matrix (not just diagonal).
        # Sigma = np.diag(self.eigenvalues)
        # self.X = rng.multivariate_normal(np.zeros(self.d), Sigma, size=self.n)

        logits = self.X @ self.w_star
        probs = 1.0 / (1.0 + np.exp(-logits))
        self.y = 2.0 * (rng.rand(self.n) < probs).astype(float) - 1.0

        print(f'Data generated: X shape = {self.X.shape}, y shape = {self.y.shape}')
        print(f'Label balance: {np.mean(self.y == 1):.2%} positive')

    # ---- Loss functions ----

    def empirical_logistic_loss(self, w):
        """Empirical logistic risk: (1/n) sum ln(1 + exp(-y_i x_i^T w))."""
        margins = self.y * (self.X @ w)
        return np.mean(np.logaddexp(0, -margins))

    def population_logistic_loss(self, w, n_samples=100000, seed=999):
        """Population logistic risk approximated via Monte Carlo."""
        rng = np.random.RandomState(seed)
        Z = rng.randn(n_samples, self.d)
        X_pop = Z * np.sqrt(self.eigenvalues)[np.newaxis, :]
        logits = X_pop @ self.w_star
        probs = 1.0 / (1.0 + np.exp(-logits))
        y_pop = 2.0 * (rng.rand(n_samples) < probs).astype(float) - 1.0
        margins = y_pop * (X_pop @ w)
        return np.mean(np.logaddexp(0, -margins))

    def logistic_gradient(self, w):
        """Gradient of the empirical logistic risk."""
        margins = self.y * (self.X @ w)
        sigmoid_neg = -1.0 / (1.0 + np.exp(margins))
        return (self.X.T @ (sigmoid_neg * self.y)) / self.n

    # ---- Gradient descent (resumable) ----

    def run_gd(self, T, log_every=2000):
        """Run (or continue) gradient descent up to step T.

        If already at step t_current, only computes steps t_current+1 to T.
        Checkpoints are log-spaced + linearly spaced for plotting.
        """
        if T <= self.t_current:
            print(f'Already at t={self.t_current}, nothing to do.')
            return

        # Initialize w if first run
        if self.w_current is None:
            self.w_current = np.zeros(self.d)
            loss = self.empirical_logistic_loss(self.w_current)
            self.w_history.append((0, self.w_current.copy()))
            self.loss_history.append((0, loss))
            print(f'  t={0:>8d}: loss={loss:.6f}, ||w||={0:.4f}')

        # Build checkpoints for the NEW range only
        log_points = set(np.unique(np.logspace(
            np.log10(max(1, self.t_current + 1)), np.log10(T), 200
        ).astype(int)))
        lin_points = set(range(
            self.t_current + log_every - (self.t_current % log_every),
            T + 1, log_every
        ))
        checkpoints_set = log_points | lin_points | {T}

        print(f'Continuing GD from t={self.t_current} to t={T}...')
        w = self.w_current.copy()

        for t in range(self.t_current + 1, T + 1):
            grad = self.logistic_gradient(w)
            w = w - self.eta * grad

            if t in checkpoints_set:
                loss = self.empirical_logistic_loss(w)
                self.w_history.append((t, w.copy()))
                self.loss_history.append((t, loss))

                if t % log_every == 0 or t == T:
                    print(f'  t={t:>8d}: loss={loss:.6f}, ||w||={norm(w):.4f}')

        self.w_current = w.copy()
        self.t_current = T
        print(f'Done. Total checkpoints: {len(self.w_history)}')

    # ---- Max-margin direction ----

    def compute_max_margin_direction(self):
        """Compute the max l2-margin direction w_tilde via dual SVM."""
        G = self.X @ self.X.T
        YGY = np.outer(self.y, self.y) * G
        n = self.n

        def dual_objective(alpha):
            return 0.5 * alpha @ YGY @ alpha - np.sum(alpha)

        def dual_gradient(alpha):
            return YGY @ alpha - np.ones(n)

        alpha0 = np.ones(n) * 0.001
        bounds = [(0, None)] * n
        result = minimize(dual_objective, alpha0, jac=dual_gradient,
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})

        alpha_star = result.x
        w_svm = self.X.T @ (alpha_star * self.y)
        self.w_tilde = w_svm / norm(w_svm)

        margins = self.y * (self.X @ w_svm)
        print(f'Max-margin SVM solved. Min margin = {np.min(margins):.6f}')
        print(f'Support vectors (alpha > 1e-6): {np.sum(alpha_star > 1e-6)}')
        return self.w_tilde

    # ---- Plotting methods ----

    def _compute_direction_stats(self):
        """Compute cosine similarities, norms from w_history."""
        w_star_dir = self.w_star / norm(self.w_star)
        times, cos_wstar, cos_wtilde, norms = [], [], [], []

        for t, w in self.w_history:
            if t == 0:
                continue
            w_dir = w / norm(w)
            times.append(t)
            cos_wstar.append(np.dot(w_dir, w_star_dir))
            if self.w_tilde is not None:
                cos_wtilde.append(np.dot(w_dir, self.w_tilde))
            norms.append(norm(w))

        return (np.array(times), np.array(cos_wstar),
                np.array(cos_wtilde) if self.w_tilde is not None else None,
                np.array(norms))

    def plot_dashboard(self, save_path=None):
        """Plot 2x2 dashboard: cosine similarities, norms, loss, angles."""
        times, cos_wstar, cos_wtilde, norms = self._compute_direction_stats()
        w_star_dir = self.w_star / norm(self.w_star)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Cosine similarity
        ax = axes[0, 0]
        ax.semilogx(times, cos_wstar,
                    label=r'$\cos(w_t/\|w_t\|,\; w^*/\|w^*\|)$', linewidth=2)
        if cos_wtilde is not None:
            ax.semilogx(times, cos_wtilde,
                        label=r'$\cos(w_t/\|w_t\|,\; \tilde{w})$', linewidth=2)
            cos_star_tilde = np.dot(w_star_dir, self.w_tilde)
            ax.axhline(cos_star_tilde, color='gray', linestyle='--', alpha=0.7,
                       label=f'cos(w*, w_tilde) = {cos_star_tilde:.4f}')
        ax.set_xlabel('GD iteration t (log scale)')
        ax.set_ylabel('Cosine similarity')
        ax.set_title('Direction of GD iterates over time')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Plot 2: Norm
        ax = axes[0, 1]
        ax.semilogx(times, norms, linewidth=2, color='green')
        ax.set_xlabel('GD iteration t (log scale)')
        ax.set_ylabel(r'$\|w_t\|$')
        ax.set_title('Norm of GD iterates over time')
        ax.grid(True, alpha=0.3)

        # Plot 3: Empirical loss
        ax = axes[1, 0]
        loss_t = [t for t, l in self.loss_history if t > 0]
        loss_v = [l for t, l in self.loss_history if t > 0]
        ax.semilogx(loss_t, loss_v, linewidth=2, color='red')
        ax.set_xlabel('GD iteration t (log scale)')
        ax.set_ylabel('Empirical logistic loss')
        ax.set_title('Empirical training loss over time')
        ax.grid(True, alpha=0.3)

        # Plot 4: Angles
        ax = axes[1, 1]
        angle_wstar = np.degrees(np.arccos(np.clip(cos_wstar, -1, 1)))
        ax.semilogx(times, angle_wstar,
                    label=r'Angle to $w^*/\|w^*\|$', linewidth=2)
        if cos_wtilde is not None:
            angle_wtilde = np.degrees(np.arccos(np.clip(cos_wtilde, -1, 1)))
            ax.semilogx(times, angle_wtilde,
                        label=r'Angle to $\tilde{w}$', linewidth=2)
        ax.set_xlabel('GD iteration t (log scale)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Angular distance from GD direction to targets')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_trajectory(self, save_path=None):
        """Plot 2D trajectory of w_t/||w_t|| projected onto (w*, w_tilde) plane."""
        if self.w_tilde is None:
            print('Run compute_max_margin_direction() first.')
            return

        w_star_dir = self.w_star / norm(self.w_star)

        # Gram-Schmidt orthonormal basis for the (w*, w_tilde) plane
        e1 = w_star_dir.copy()
        e2 = self.w_tilde - np.dot(self.w_tilde, e1) * e1
        e2 = e2 / norm(e2)

        proj_wstar = np.array([np.dot(w_star_dir, e1), np.dot(w_star_dir, e2)])
        proj_wtilde = np.array([np.dot(self.w_tilde, e1), np.dot(self.w_tilde, e2)])

        proj_traj, traj_times = [], []
        for t, w in self.w_history:
            if t == 0:
                continue
            w_dir = w / norm(w)
            proj_traj.append([np.dot(w_dir, e1), np.dot(w_dir, e2)])
            traj_times.append(t)

        proj_traj = np.array(proj_traj)
        traj_times = np.array(traj_times)

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(proj_traj[:, 0], proj_traj[:, 1],
                             c=np.log10(traj_times), cmap='viridis', s=15, zorder=2)
        ax.plot(proj_traj[:, 0], proj_traj[:, 1],
                color='gray', alpha=0.3, linewidth=0.5, zorder=1)

        ax.scatter(*proj_wstar, color='red', s=200, marker='*',
                   zorder=5, label=r'$w^*/\|w^*\|$')
        ax.scatter(*proj_wtilde, color='blue', s=200, marker='D',
                   zorder=5, label=r'$\tilde{w}$ (max margin)')
        ax.scatter(proj_traj[0, 0], proj_traj[0, 1], color='green', s=100,
                   marker='o', zorder=5, label=f't={traj_times[0]} (start)')
        ax.scatter(proj_traj[-1, 0], proj_traj[-1, 1], color='black', s=100,
                   marker='s', zorder=5, label=f't={traj_times[-1]} (end)')

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log10(t)')
        ax.set_xlabel(r'Projection onto $w^*/\|w^*\|$ direction')
        ax.set_ylabel(r'Projection onto orthogonal direction')
        ax.set_title(r'Trajectory of $w_t/\|w_t\|$ projected onto $(w^*, \tilde{w})$ plane')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    def print_summary(self, key_times=None):
        """Print summary table at key time points."""
        w_star_dir = self.w_star / norm(self.w_star)

        if key_times is None:
            key_times = sorted(set(
                [1, 10, 50, 100, 500, 1000, 2000, 5000] +
                [10**i for i in range(1, int(np.log10(max(1, self.t_current))) + 1)] +
                [self.t_current]
            ))
            key_times = [t for t in key_times if t <= self.t_current]

        has_wtilde = self.w_tilde is not None
        header = f"{'t':>10s} | {'cos(w_t, w*)':>14s}"
        if has_wtilde:
            header += f" | {'cos(w_t, w_tilde)':>18s}"
        header += f" | {'||w_t||':>10s} | {'emp. loss':>10s}"
        print(header)
        print('-' * len(header))

        for t, w in self.w_history:
            if t in key_times and t > 0:
                w_dir = w / norm(w)
                cs = np.dot(w_dir, w_star_dir)
                wn = norm(w)
                loss = self.empirical_logistic_loss(w)
                row = f'{t:>10d} | {cs:>14.6f}'
                if has_wtilde:
                    ct = np.dot(w_dir, self.w_tilde)
                    row += f' | {ct:>18.6f}'
                row += f' | {wn:>10.4f} | {loss:>10.6f}'
                print(row)
