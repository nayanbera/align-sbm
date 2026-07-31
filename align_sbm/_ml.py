"""Lightweight Gaussian Process Regressor — numpy/scipy only, no sklearn needed."""
import numpy as np

try:
    from scipy.optimize import minimize as _minimize
    _SCIPY_OPT = True
except ImportError:
    _SCIPY_OPT = False


class NumpyGPR:
    """
    Gaussian Process Regressor with an RBF (squared-exponential) kernel + noise.

    Hyperparameters (length_scale, output_scale, noise_level) are optimised by
    maximising the log-marginal-likelihood via L-BFGS-B with multiple restarts.
    Both X and y are internally standardised for numerical stability.
    """

    def __init__(self):
        self._fitted       = False
        self.length_scale_ = 1.0
        self.output_scale_ = 1.0
        self.noise_        = 1e-3

    # ── Kernel ────────────────────────────────────────────────────────────────

    @staticmethod
    def _rbf(X1, X2, ls, os):
        d = (X1[:, None] - X2[None, :]) / ls
        return (os * os) * np.exp(-0.5 * d * d)

    # ── Negative log-marginal-likelihood (minimise) ───────────────────────────

    def _neg_lml(self, log_params, Xs, ys):
        ls = np.exp(log_params[0])
        os = np.exp(log_params[1])
        sn = np.exp(log_params[2])
        n  = len(ys)
        K  = self._rbf(Xs, Xs, ls, os) + (sn * sn + 1e-10) * np.eye(n)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, ys))
        lml   = (-0.5 * (ys @ alpha)
                 - np.sum(np.log(np.diag(L)))
                 - 0.5 * n * np.log(2.0 * np.pi))
        return -float(lml)

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """Train on (X, y); X must be 1-D (MonoE values)."""
        X = np.asarray(X, float).ravel()
        y = np.asarray(y, float)

        self._Xmean = X.mean();  self._Xstd = X.std() or 1.0
        self._ymean = y.mean();  self._ystd = y.std() or 1.0
        Xs = (X - self._Xmean) / self._Xstd
        ys = (y - self._ymean) / self._ystd
        self._Xs = Xs

        if _SCIPY_OPT and len(X) >= 2:
            best_val, best_x = np.inf, np.array([0.0, 0.0, -3.0])
            for x0 in [
                np.array([ 0.0,  0.0, -3.0]),
                np.array([-0.5,  0.5, -2.0]),
                np.array([ 1.0, -0.5, -4.0]),
                np.array([-1.0, -1.0, -1.0]),
            ]:
                res = _minimize(
                    self._neg_lml, x0, args=(Xs, ys),
                    method="L-BFGS-B",
                    bounds=[(-5, 5), (-5, 5), (-8, 1)],
                    options={"maxiter": 400, "ftol": 1e-10},
                )
                if res.fun < best_val:
                    best_val, best_x = res.fun, res.x
            log_ls, log_os, log_sn = best_x
        else:
            log_ls, log_os, log_sn = 0.0, 0.0, -3.0

        self.length_scale_ = float(np.exp(log_ls))
        self.output_scale_ = float(np.exp(log_os))
        self.noise_        = float(np.exp(log_sn))

        n  = len(Xs)
        K  = self._rbf(Xs, Xs, self.length_scale_, self.output_scale_)
        K += (self.noise_ * self.noise_ + 1e-10) * np.eye(n)
        self._L     = np.linalg.cholesky(K)
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, ys))
        self._ys    = ys
        self._fitted = True
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, X_new, return_std=False):
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        X_new  = np.asarray(X_new, float).ravel()
        Xs_new = (X_new - self._Xmean) / self._Xstd

        K_star = self._rbf(Xs_new, self._Xs,
                            self.length_scale_, self.output_scale_)
        mu_s   = K_star @ self._alpha
        mu     = mu_s * self._ystd + self._ymean

        if not return_std:
            return mu

        v      = np.linalg.solve(self._L, K_star.T)
        K_ss   = self._rbf(Xs_new, Xs_new,
                            self.length_scale_, self.output_scale_)
        var_s  = np.diag(K_ss) - np.sum(v * v, axis=0)
        std    = np.sqrt(np.maximum(var_s, 0.0)) * self._ystd
        return mu, std

    def score(self, X, y):
        y  = np.asarray(y, float)
        yp = self.predict(np.asarray(X, float))
        ss_res = float(np.sum((y - yp) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # ── Leave-one-out cross-validation ────────────────────────────────────────

    def loo_r2(self, X, y):
        """LOO cross-validated R². Trains n separate models (fast for n ≤ 30)."""
        X = np.asarray(X, float).ravel()
        y = np.asarray(y, float)
        n = len(X)
        if n < 3:
            return float("nan")
        loo_preds = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, bool)
            mask[i] = False
            try:
                g = NumpyGPR().fit(X[mask], y[mask])
                loo_preds[i] = float(g.predict(X[[i]])[0])
            except Exception:
                loo_preds[i] = y[mask].mean()
        ss_res = float(np.sum((y - loo_preds) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
