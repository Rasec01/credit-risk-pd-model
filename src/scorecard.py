"""
Logistic-regression scorecard, built the way credit-risk teams build them:

1. Bin each ratio into a few ranges (quantiles of the development sample;
   missing values get their own bin).
2. Replace each bin by its weight of evidence: WoE = ln(% of non-defaults / % of defaults).
   High WoE = safer than average, low WoE = riskier.
3. Keep ratios with enough information value (IV) and drop near-duplicates.
4. Fit a logistic regression on the WoE values.
5. Turn the model into points: 600 points at odds of 50:1, and 20 more
   points every time the odds of not defaulting double.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MISSING = "missing"


class WoEBinner:
    def __init__(self, n_bins: int = 5, smoothing: float = 0.5):
        self.n_bins = n_bins
        self.smoothing = smoothing  # avoids ln(0) when a bin has no defaults

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.edges_, self.woe_, rows = {}, {}, []
        n_bad, n_good = y.sum(), (1 - y).sum()
        for col in X.columns:
            x = X[col]
            q = np.unique(np.nanquantile(x, np.linspace(0, 1, self.n_bins + 1)))
            if len(q) < 3:  # fewer than two distinct bins: nothing to learn
                continue
            edges = np.concatenate([[-np.inf], q[1:-1], [np.inf]])
            bins = self._assign(x, edges)
            table = pd.DataFrame({"bin": bins, "bad": y}).groupby("bin")["bad"].agg(["sum", "count"])
            bad = table["sum"] + self.smoothing
            good = table["count"] - table["sum"] + self.smoothing
            dist_bad, dist_good = bad / (n_bad + self.smoothing), good / (n_good + self.smoothing)
            woe = np.log(dist_good / dist_bad)
            iv = float(((dist_good - dist_bad) * woe).sum())
            self.edges_[col], self.woe_[col] = edges, woe.to_dict()
            rows.append({"feature": col, "iv": iv, "bins": len(table)})
        self.iv_ = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
        return self

    @staticmethod
    def _assign(x: pd.Series, edges) -> pd.Series:
        labels = pd.cut(x, edges, right=True).astype(str)
        return labels.where(x.notna(), MISSING)

    def transform(self, X: pd.DataFrame, features=None) -> pd.DataFrame:
        out = {}
        for col in features or list(self.edges_):
            bins = self._assign(X[col], self.edges_[col])
            # A bin never seen in development (e.g. missing) gets WoE 0, i.e. average risk
            out[col] = bins.map(self.woe_[col]).fillna(0.0).astype(float)
        return pd.DataFrame(out, index=X.index)


def select_features(binner: WoEBinner, X_woe: pd.DataFrame, min_iv=0.1, max_corr=0.7, max_features=10):
    """Highest-IV first; skip a ratio if it is too correlated with one already chosen."""
    chosen = []
    for col in binner.iv_.loc[binner.iv_["iv"] >= min_iv, "feature"]:
        if all(abs(X_woe[col].corr(X_woe[c])) <= max_corr for c in chosen):
            chosen.append(col)
        if len(chosen) == max_features:
            break
    return chosen


class Scorecard:
    def __init__(self, n_bins=5, min_iv=0.1, max_corr=0.7, max_features=10,
                 base_score=600, base_odds=50, pdo=20):
        self.binner = WoEBinner(n_bins)
        self.min_iv, self.max_corr, self.max_features = min_iv, max_corr, max_features
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    def fit(self, X: pd.DataFrame, y):
        self.binner.fit(X, y)
        X_woe = self.binner.transform(X)
        self.features_ = select_features(self.binner, X_woe, self.min_iv, self.max_corr, self.max_features)
        if not self.features_:
            raise ValueError("No ratio reaches the minimum information value.")
        # A scorecard variable must push the score the right way: higher WoE (safer bin) has to lower
        # the PD. A positive coefficient means the ratio is mostly repeating information already carried
        # by a correlated variable, so it is dropped and the model refitted until every sign is right.
        self.dropped_ = []
        while True:
            self.model_ = LogisticRegression(C=1.0, max_iter=1000).fit(X_woe[self.features_], y)
            coef = self.model_.coef_[0]
            if coef.max() <= 0 or len(self.features_) == 1:
                return self
            self.dropped_.append(self.features_.pop(int(coef.argmax())))

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self.binner.transform(X, self.features_))[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Points: higher = safer. ln(odds of not defaulting) is minus the model's logit."""
        logit = self.model_.decision_function(self.binner.transform(X, self.features_))
        return self.offset - self.factor * logit

    def points_table(self) -> pd.DataFrame:
        """Points per bin, so a credit officer can add up a firm's score by hand."""
        n = len(self.features_)
        a = self.model_.intercept_[0]
        rows = []
        for col, b in zip(self.features_, self.model_.coef_[0]):
            for bin_label, woe in self.binner.woe_[col].items():
                pts = self.offset / n - self.factor * (b * woe + a / n)
                rows.append({"feature": col, "bin": bin_label, "woe": round(woe, 3), "points": round(pts)})
        return pd.DataFrame(rows)

    def coefficients(self) -> pd.DataFrame:
        iv = self.binner.iv_.set_index("feature")["iv"]
        return pd.DataFrame({"feature": self.features_, "coefficient": self.model_.coef_[0],
                             "iv": [iv[f] for f in self.features_]})
