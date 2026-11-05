"""Scoring harness shared by the modelling notebooks.

Tiers (the original framing): a prediction is a tier distribution per vote; its score is the information it carries
about the measured tier, H(T) - log-loss, after shrinking the distribution toward uniform by a λ chosen on validation
(floor 0.005, so no tier can get probability zero). Hard-tier accuracies from the most probable tier are reported alongside.

Continuous (the second framing, see 11_continuous.ipynb): a prediction is a point estimate of logS per compound,
turned into a density over the measured value by the replicate noise density (a symmetrized KDE of the residuals from
the prep notebook) widened by the model's own error τ and mixed with a wide gross-error component ε, both chosen on
validation. Its score is h(Y) - cross-entropy of that density on the held-out votes, in bits.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d

STEP, HALF = 0.01, 30.0  # logS grid the densities live on, as in the prep notebook's floor
TAUS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0)  # model-error SD added to the noise (logS), tried on validation
EPSILONS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2)  # share of a wide gross-error component, tried on validation
WIDE_SD = 2.4  # the gross-error scale from the prep notebook (flagged votes sit ~2 logS from consensus)
METRICS_C = {"information (bits)": 3, "RMSE (logS)": 3, "MAE (logS)": 3, "within 0.5 logS (%)": 1, "τ (logS)": 2, "ε": 2}

LAMBDAS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)  # shrinkage toward uniform tried on validation; floor > 0
METRICS = {  # decimals shown
    "information (bits)": 3,
    "share of tier entropy (%)": 1,
    "accuracy: same tier (%)": 1,
    "1 tier off (%)": 1,
    "2+ tiers off (%)": 1,
}


def entropy(p: np.ndarray) -> float:
    """Entropy (bits) of a distribution."""
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def information(measured: np.ndarray, proba: np.ndarray, weight: np.ndarray, lam: float) -> tuple[float, float]:
    """H(T) - log-loss of the shrunk probabilities, and H(T), both in bits, vote-weighted."""
    tiers = proba.shape[1]
    q = (1 - lam) * proba + lam / tiers
    h = entropy(np.bincount(measured, weights=weight, minlength=tiers) / weight.sum())
    log_loss = -(weight * np.log2(q[np.arange(len(measured)), measured])).sum() / weight.sum()
    return h - log_loss, h


def calibrate(measured: np.ndarray, proba: np.ndarray, weight: np.ndarray, lambdas=LAMBDAS) -> float:
    """The shrinkage toward uniform with the lowest log-loss on these (validation) rows."""
    return max(lambdas, key=lambda lam: information(measured, proba, weight, lam)[0])


def score(measured: np.ndarray, proba: np.ndarray, weight: np.ndarray, lam: float) -> dict:
    """Information of the calibrated prediction, plus where the most probable tier lands relative to the measured one."""
    tiers = proba.shape[1]
    info, h = information(measured, proba, weight, lam)
    apart = np.abs(np.subtract.outer(range(tiers), range(tiers)))
    p = np.zeros((tiers, tiers))
    np.add.at(p, (measured, proba.argmax(axis=1)), weight)
    p /= p.sum()
    return {
        "information (bits)": info,
        "share of tier entropy (%)": 100 * info / h,
        "accuracy: same tier (%)": 100 * p[apart == 0].sum(),
        "1 tier off (%)": 100 * p[apart == 1].sum(),
        "2+ tiers off (%)": 100 * p[apart >= 2].sum(),
    }


def corrected_se(values: pd.Series, n_splits: int, test_to_train: float) -> float:
    """Standard error of a mean over cross-validation splits, corrected for overlapping training sets (Nadeau & Bengio)."""
    return float(np.sqrt((1 / n_splits + test_to_train) * values.var()))


def formatted(df: pd.DataFrame, metrics: dict = METRICS) -> pd.DataFrame:
    """Each metric as text, with its own number of decimals."""
    return pd.DataFrame({m: df[m].map(f"{{:.{d}f}}".format) for m, d in metrics.items() if m in df}, index=df.index)


# ---- continuous target ------------------------------------------------------------------------------------------

GRID = np.arange(-HALF, HALF + STEP, STEP)


def differential_entropy(density: np.ndarray) -> float:
    """h of a density tabulated on GRID, in bits."""
    f = density[density > 0]
    return float(-(f * np.log2(f)).sum() * STEP)


def noise_residuals(measurements: pd.DataFrame) -> np.ndarray:
    """Each unflagged vote's distance from the mean of its compound's other unflagged votes (3+ good votes), scaled by
    sqrt((K-1)/K) so its variance matches a single measurement's. Exactly the prep notebook's residuals."""
    out = []
    for _, good in measurements[~measurements["flagged"]].groupby("InChIKey")["Solubility"]:
        g = good.to_numpy()
        if len(g) >= 3:
            out += [(g[k] - np.delete(g, k).mean()) * np.sqrt((len(g) - 1) / len(g)) for k in range(len(g))]
    return np.array(out)


class Noise:
    """The replicate noise density e (symmetrized KDE on GRID) and the predictive densities built from it."""

    def __init__(self, residuals: np.ndarray):
        fe = stats.gaussian_kde(np.concatenate([residuals, -residuals]))(GRID)
        self.density = fe / (fe.sum() * STEP)
        self.entropy = differential_entropy(self.density)  # h(e), bits
        self._cache = {}

    def predictive(self, tau: float, eps: float) -> np.ndarray:
        """Density of (vote - prediction): noise ⊛ N(0, τ²), mixed with ε of N(0, WIDE_SD²)."""
        if (tau, eps) not in self._cache:
            f = gaussian_filter1d(self.density, tau / STEP, mode="constant") if tau > 0 else self.density
            self._cache[tau, eps] = np.maximum((1 - eps) * f + eps * stats.norm.pdf(GRID, 0, WIDE_SD), 1e-300)  # no log of zero in the far tails
        return self._cache[tau, eps]

    def log2_density(self, residual: np.ndarray, tau: float, eps: float) -> np.ndarray:
        return np.log2(np.interp(residual, GRID, self.predictive(tau, eps), left=1e-300, right=1e-300))

    def interval(self, tau: float, eps: float, coverage: float = 0.9) -> float:
        """Half-width of the central `coverage` interval of the predictive density (logS)."""
        cdf = np.cumsum(self.predictive(tau, eps)) * STEP
        return float(np.interp((1 + coverage) / 2, cdf, GRID))


def entropy_c(y: np.ndarray, weight: np.ndarray) -> float:
    """h(Y) of the measured values (weighted KDE), bits."""
    return differential_entropy(stats.gaussian_kde(y, weights=weight)(GRID))


def information_c(y: np.ndarray, prediction: np.ndarray, weight: np.ndarray, noise: Noise, tau: float, eps: float,
                  h: float | None = None) -> tuple[float, float]:
    """h(Y) - cross-entropy of the predictive density on these votes, and h(Y), both in bits, vote-weighted."""
    if h is None:
        h = entropy_c(y, weight)
    cross = -(weight * noise.log2_density(y - prediction, tau, eps)).sum() / weight.sum()
    return h - cross, h


def calibrate_c(y: np.ndarray, prediction: np.ndarray, weight: np.ndarray, noise: Noise) -> tuple[float, float]:
    """The (τ, ε) with the lowest cross-entropy on these (validation) rows."""
    h = entropy_c(y, weight)
    return max(((t, e) for t in TAUS for e in EPSILONS), key=lambda te: information_c(y, prediction, weight, noise, *te, h=h)[0])


def score_c(y: np.ndarray, prediction: np.ndarray, weight: np.ndarray, noise: Noise, tau: float, eps: float) -> dict:
    """Information of the calibrated predictive density, plus how far the point prediction lands from the vote."""
    info, _ = information_c(y, prediction, weight, noise, tau, eps)
    err = y - prediction
    return {
        "information (bits)": info,
        "RMSE (logS)": float(np.sqrt((weight * err**2).sum() / weight.sum())),
        "MAE (logS)": float((weight * np.abs(err)).sum() / weight.sum()),
        "within 0.5 logS (%)": float(100 * (weight * (np.abs(err) <= 0.5)).sum() / weight.sum()),
        "τ (logS)": tau,
        "ε": eps,
    }


if __name__ == "__main__":  # self-check: the smallest things that fail if the scoring breaks
    rng = np.random.default_rng(0)
    # continuous: a perfect predictor of the mean scores h(Y) - h(e) with τ = 0; the marginal mean scores ~0; τ absorbs model error
    sd, n = 0.5, 4000
    mu = rng.normal(-3, 2, n)
    noise = Noise(rng.normal(0, sd, 3000))
    y, w = mu + rng.normal(0, sd, n), np.ones(n)
    tau, eps = calibrate_c(y, mu, w, noise)
    assert tau <= 0.1 and eps == 0, (tau, eps)  # τ = 0.1 is within the sampling noise of a 3000-residual KDE
    info = information_c(y, mu, w, noise, tau, eps)[0]
    floor = 0.5 * np.log2(1 + 4 / sd**2)
    assert abs(info - floor) < 0.1, (info, floor)
    assert abs(information_c(y, np.full(n, y.mean()), w, noise, *calibrate_c(y, np.full(n, y.mean()), w, noise))[0]) < 0.05
    noisy = mu + rng.normal(0, 1.0, n)  # a model with its own error SD of 1
    tau, eps = calibrate_c(y, noisy, w, noise)
    assert tau == 1.0 and eps == 0, (tau, eps)
    assert 0.78 < noise.interval(0, 0) < 0.9, noise.interval(0, 0)  # 90% of N(0, 0.5²) lies within ±0.82
    print("continuous self-check passed")
    measured = rng.integers(0, 4, size=2000)
    weight = np.ones(2000)
    h = entropy(np.bincount(measured, minlength=4) / 2000)
    perfect = np.eye(4)[measured]
    marginal = np.tile(np.bincount(measured, minlength=4) / 2000, (2000, 1))
    assert abs(information(measured, marginal, weight, 0.005)[0]) < 1e-3, "predicting the marginal carries no information"
    assert 0 < h - information(measured, perfect, weight, 0.005)[0] < 0.01, "a perfect prediction carries almost all of H(T)"
    wrong = np.eye(4)[(measured + 2) % 4]
    assert information(measured, wrong, weight, 0.005)[0] < -5, "a confidently wrong prediction has negative information"
    assert calibrate(measured, wrong, weight) == max(LAMBDAS), "calibration hedges a wrong predictor as far as it can"
    s = score(measured, perfect, weight, 0.005)
    assert s["accuracy: same tier (%)"] == 100 and s["2+ tiers off (%)"] == 0
    print("harness self-check passed")
