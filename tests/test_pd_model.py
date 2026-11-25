import numpy as np
import pandas as pd
import pytest

from src import validation as v
from src.data import clean_name, split_data
from src.scorecard import Scorecard, WoEBinner


def fake_firms(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "roa": rng.normal(0.05, 0.08, n),
        "debt_ratio": rng.uniform(0.1, 0.9, n),
        "current_ratio": rng.lognormal(0.3, 0.4, n),
        "noise": rng.normal(0, 1, n),
    })
    df.loc[rng.choice(n, 200, replace=False), "current_ratio"] = np.nan
    logit = -3.5 - 12 * df["roa"] + 3 * df["debt_ratio"]
    df["default"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture(scope="module")
def fitted():
    X_dev, X_test, y_dev, y_test = split_data(fake_firms())
    return Scorecard(max_features=5).fit(X_dev, y_dev), X_dev, X_test, y_dev, y_test


def test_clean_name():
    assert clean_name(" ROA(C) before interest ") == "roa_c_before_interest"


def test_woe_sign_and_iv():
    x = pd.DataFrame({"risky_when_high": np.arange(1000.0)})
    y = (x["risky_when_high"] > 800).astype(int)
    b = WoEBinner(n_bins=5).fit(x, y)
    woe = list(b.woe_["risky_when_high"].values())
    assert woe[0] > 0 > woe[-1]              # low values safe, high values risky
    assert b.iv_.loc[0, "iv"] > 0.5         # a perfect separator carries a lot of information


def test_bins_come_from_development_data_only():
    b = WoEBinner().fit(pd.DataFrame({"a": np.arange(100.0)}), np.r_[np.zeros(90), np.ones(10)])
    edges = b.edges_["a"].copy()
    b.transform(pd.DataFrame({"a": [1e6, -1e6]}))
    np.testing.assert_array_equal(edges, b.edges_["a"])


def test_scorecard_picks_real_drivers_with_negative_coefficients(fitted):
    card = fitted[0]
    assert {"roa", "debt_ratio"} <= set(card.features_)
    assert (card.coefficients()["coefficient"] < 0).all()


def test_points_add_up_to_the_score(fitted):
    card, _, X_test, _, _ = fitted
    table = card.points_table()
    row = X_test.iloc[[0]]
    total = 0
    for f in card.features_:
        label = card.binner._assign(row[f], card.binner.edges_[f]).iloc[0]
        total += table.query("feature == @f and bin == @label")["points"].iloc[0]
    assert total == pytest.approx(card.score(row)[0], abs=len(card.features_))  # rounding per bin


def test_higher_score_means_lower_pd(fitted):
    card, _, X_test, _, _ = fitted
    s, p = pd.Series(card.score(X_test)), pd.Series(card.predict_pd(X_test))
    assert s.corr(p, method="spearman") == pytest.approx(-1.0)  # same ranking, opposite direction


def test_validation_measures(fitted):
    card, X_dev, X_test, y_dev, y_test = fitted
    p = card.predict_pd(X_test)
    assert v.discrimination(y_test, p)["gini"] > 0.3
    assert v.psi(card.score(X_dev), card.score(X_dev)) == pytest.approx(0, abs=1e-9)
    assert v.psi(np.arange(1000.0), np.arange(1000.0) + 500) > 0.25
    grades = v.rating_grades(card.score(X_dev), card.score(X_test), y_test)
    assert grades.index.min() == 1 and grades["default_rate"].iloc[-1] > grades["default_rate"].iloc[0]


def test_hosmer_lemeshow_flags_bad_calibration():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.01, 0.2, 5000)
    y = (rng.random(5000) < p).astype(int)
    assert v.hosmer_lemeshow(y, p)["hl_p_value"] > 0.01
    assert v.hosmer_lemeshow(y, np.clip(p * 3, 0, 0.99))["hl_p_value"] < 0.01


def test_full_run_writes_the_report(tmp_path, monkeypatch):
    import sys
    import main
    csv = tmp_path / "firms.csv"
    fake_firms(3000).to_csv(csv, index=False)
    monkeypatch.setattr(sys, "argv", ["main.py", "--data", str(csv), "--out", str(tmp_path / "out")])
    main.main()
    report = (tmp_path / "out" / "validation_report.md").read_text()
    assert "## Findings" in report and "Gini" in report
    assert (tmp_path / "out" / "points_table.csv").exists()


def test_wrong_signed_variables_are_dropped():
    # 'b' is safer on its own but riskier once 'a' is known: a classic suppressor that would
    # get a positive coefficient, so the scorecard has to drop it.
    rng = np.random.default_rng(5)
    f, u = rng.normal(0, 1, 8000), rng.normal(0, 1, 8000)
    df = pd.DataFrame({"a": f + 0.6 * u, "b": f})
    y = (rng.random(8000) < 1 / (1 + np.exp(3 + 2 * u + 0.5 * f))).astype(int)
    card = Scorecard(max_corr=1.0).fit(df, y)
    assert card.dropped_ == ["b"]
    assert (card.coefficients()["coefficient"] <= 0).all()
