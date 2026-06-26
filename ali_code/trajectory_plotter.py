import numpy as np


class TrajectoryValidationRiskPloter:
    def __init__(self, max_T=1e3, test_samples=1e5, random_seed=42):
        self.X = None
        self.y = None
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.w_traj = None
        self.train_over_time = None
        self.valid_over_time = None
        self.test_over_time = None
        self.beta_0 = None
        self.noise_sigma = None
        self.isotropic = True
        self.rng = np.random.default_rng(random_seed)
        self.max_T = max_T

    def set_max_T(self, T):
        self.max_T = T

    def split_train_valid(self, ratio=0.5):
        n = len(self.X)
        training_size = int(n * ratio)
        X_train = self.X[0:training_size, :]
        y_train = self.y[0:training_size]
        X_valid = self.X[training_size:, :]
        y_valid = self.y[training_size:]
        return X_train, y_train, X_valid, y_valid

    def generate_data(self, n=3000, p=6000, paper="patil24a",
                      isotropic=True, Sigma=None,
                      beta_0=None, noise_sigma=1):
        if beta_0 is None:
            beta_0 = 5 * np.identity(p)[0]
        self.beta_0 = beta_0

        if isotropic:
            self.isotropic = True
            Sigma = np.identity(p)
        else:
            if Sigma is None:
                raise ValueError("Sigma is not specified")
        self.Sigma = Sigma

        X = self.rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)
        y = X @ beta_0 + self.rng.normal(0, noise_sigma, n)
        return X, y

    def run_GD_efficient(self, X_train, y_train, X_valid, y_valid, X_test, y_test,
                         eta=0.1, w_0=None):
        if w_0 is None:
            w_0 = np.zeros(len(X_train[0]))
        n = len(X_train)
        w_prev = w_0
        training_error_over_time = []
        validation_error_over_time = []
        test_error_over_time = []
        for t in range(1, int(self.max_T) + 1):
            w_t = w_prev + (eta / n) * X_train.T @ (y_train - X_train @ w_prev)
            w_prev = w_t
            training_error_over_time.append(self.evaluate(w_t, X_train, y_train))
            validation_error_over_time.append(self.evaluate(w_t, X_valid, y_valid))
            test_error_over_time.append(self.evaluate(w_t, X_test, y_test))
        return None, training_error_over_time, validation_error_over_time, test_error_over_time

    def evaluate(self, w, X, y):
        n = len(X)
        r = y - X @ w
        return np.dot(r, r) / (2 * n)

    def run_the_whole_thing(self, n=3000, p=6000, random_seed=42, test_sample_size=1e4):
        self.X, self.y = self.generate_data(n=n, p=p)
        X_train, y_train, X_valid, y_valid = self.split_train_valid()
        X_test, y_test = self.generate_data(n=int(test_sample_size), p=p)
        _, self.train_over_time, self.valid_over_time, self.test_over_time = self.run_GD_efficient(
            X_train, y_train, X_valid, y_valid, X_test, y_test
        )
