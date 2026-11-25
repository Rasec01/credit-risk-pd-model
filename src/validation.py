"""
The checks a model-validation team runs on a PD model.

Rules of thumb used in the findings (common in practice, not regulation):
Gini >= 0.6 strong, 0.4-0.6 acceptable, < 0.4 weak;
PSI < 0.10 stable, 0.10-0.25 some shift, > 0.25 significant shift.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def discrimination(y, pd_hat) -> dict:
    auc = roc_auc_score(y, pd_hat)
    fpr, tpr, _ = roc_curve(y, pd_hat)
    return {"auc": auc, "gini": 2 * auc - 1, "ks": float(np.max(tpr - fpr))}


def calibration_table(y, pd_hat, groups: int = 10) -> pd.DataFrame:
    """Predicted vs observed default rate in groups of equal size, riskiest last."""
    df = pd.DataFrame({"y": y, "pd": pd_hat})
    df["group"] = pd.qcut(df["pd"].rank(method="first"), groups, labels=False) + 1
    return df.groupby("group").agg(firms=("y", "size"), predicted=("pd", "mean"), observed=("y", "mean"))


def hosmer_lemeshow(y, pd_hat, groups: int = 10) -> dict:
    """Large statistic / small p-value = predicted PDs do not match observed default rates."""
    t = calibration_table(y, pd_hat, groups)
    expected = t["predicted"] * t["firms"]
    observed = t["observed"] * t["firms"]
    var = t["firms"] * t["predicted"] * (1 - t["predicted"])
    stat = float(((observed - expected) ** 2 / var.clip(lower=1e-9)).sum())
    return {"hl_statistic": stat, "hl_p_value": float(1 - chi2.cdf(stat, groups - 2))}


def psi(expected_scores, actual_scores, bins: int = 10) -> float:
    """Population stability index of the score between two samples; bins from the first."""
    edges = np.unique(np.quantile(expected_scores, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected_scores, edges)[0] / len(expected_scores)
    a = np.histogram(actual_scores, edges)[0] / len(actual_scores)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def rating_grades(dev_scores, scores, y, n_grades: int = 7) -> pd.DataFrame:
    """
    Grades 1 (safest) to n from score bands set on the development sample.
    Default rates should rise steadily from grade 1 to grade n.
    """
    cuts = np.quantile(dev_scores, np.linspace(0, 1, n_grades + 1))[1:-1]
    grade = n_grades - np.searchsorted(cuts, scores, side="right")
    t = pd.DataFrame({"grade": grade, "y": y}).groupby("grade")["y"].agg(firms="size", defaults="sum")
    t["default_rate"] = t["defaults"] / t["firms"]
    return t


def findings(stats: dict, grades: pd.DataFrame) -> list[str]:
    out = []
    g = stats["gini"]
    out.append(f"Discrimination is {'strong' if g >= 0.6 else 'acceptable' if g >= 0.4 else 'weak'} "
               f"(Gini {g:.2f}).")
    p = stats["psi"]
    out.append(f"The score distribution is {'stable' if p < 0.1 else 'somewhat shifted' if p < 0.25 else 'significantly shifted'} "
               f"between development and test samples (PSI {p:.3f}).")
    if stats["hl_p_value"] < 0.05:
        out.append(f"Calibration fails the Hosmer-Lemeshow test (p = {stats['hl_p_value']:.3f}): "
                   "predicted PDs should be recalibrated before use.")
    else:
        out.append(f"Calibration passes the Hosmer-Lemeshow test (p = {stats['hl_p_value']:.3f}).")
    rates = grades["default_rate"].to_numpy()
    if np.all(np.diff(rates) >= 0):
        out.append("Default rates rise monotonically from the safest to the riskiest grade.")
    else:
        out.append("Default rates are not monotonic across grades; some grade boundaries should be merged.")
    return out
