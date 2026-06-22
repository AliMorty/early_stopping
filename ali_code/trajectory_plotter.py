import numpy as np


class TrajectoryValidationRiskPloter:
    def __init__(self, max_T=5e3):
        self.X = None
        self.y = None

        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None

        self.w_traj = None
        self.train_over_time = None
        self.valid_over_time = None

        self.max_T = max_T

    def set_max_T(self, T):
        self.max_T = T

    def generate_data(self, n=3000, p=6000, paper="patil24a",
                      isotropic=True,
                      Sigma=None, random_seed=42,
                      beta_0=None, noise_sigma=1):

        p += 1  # to keep p+1 dimensions as stated in the paper

        rng = np.random.default_rng(random_seed)

        if beta_0 is None:
            beta_0 = 5 * np.identity(p)[0]

        if isotropic:
            Sigma = np.identity(p)
        X = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)
        y = X @ beta_0 + rng.normal(0, noise_sigma, n)
        return X, y

    def split_train_valid(self, ratio=0.5):
        n = len(self.X)
        training_size = int(n * ratio)
        X_train = self.X[0:training_size, :]
        y_train = self.y[0:training_size]
        X_valid = self.X[training_size:, :]
        y_valid = self.y[training_size:]
        return X_train, y_train, X_valid, y_valid

    def run_GD(self, X_train, y_train, X_valid, y_valid, eta=0.1, w_0=None):
        if w_0 is None:
            w_0 = np.zeros(len(X_train[0]))
        n = len(X_train)

        trajectory = [w_0]
        training_error_over_time = []
        validation_error_over_time = []

        for t in range(1, int(self.max_T)):
            w_prev = trajectory[-1]
            w_t = w_prev + (eta / n) * X_train.T @ (y_train - X_train @ w_prev)
            trajectory.append(w_t)
            training_error_over_time.append(self.evaluate(w_t, X_train, y_train))
            validation_error_over_time.append(self.evaluate(w_t, X_valid, y_valid))

        return trajectory, training_error_over_time, validation_error_over_time

    def evaluate(self, w, X, y):
        n = len(X)
        r = y - X @ w
        return np.dot(r, r) / (2 * n)

    def run_the_whole_thing(self, n=3000, p=6000, random_seed=42):
        self.X, self.y = self.generate_data(n=n, p=p, random_seed=random_seed)
        X_train, y_train, X_valid, y_valid = self.split_train_valid()
        self.w_traj, self.train_over_time, self.valid_over_time = self.run_GD(
            X_train, y_train, X_valid, y_valid
        )
