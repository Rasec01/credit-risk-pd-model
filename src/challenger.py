"""
Challenger: gradient boosting on the raw ratios. It handles missing values and
non-linear effects by itself, so it often ranks firms better than the scorecard;
the question for validation is whether that gain is worth losing transparency.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def train_challenger(X, y, seed: int = 42):
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.2, random_state=seed)
    return model.fit(X, y)


def shap_values(model, X, max_rows: int = 1000):
    """SHAP values on a sample of rows; None if shap is not installed."""
    try:
        import shap
    except ImportError:
        return None, None
    sample = X.iloc[:max_rows]
    values = shap.TreeExplainer(model).shap_values(sample)
    values = values[1] if isinstance(values, list) else values
    return (values[:, :, 1] if np.ndim(values) == 3 else values), sample
