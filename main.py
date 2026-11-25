"""
Credit risk PD model: scorecard, challenger and validation report.

    python scripts/download_data.py     # public data -> data/companies.csv
    python main.py                      # writes outputs/validation_report.md and charts
"""
import argparse
from pathlib import Path

from sklearn.metrics import brier_score_loss

from src import validation as v
from src.challenger import shap_values, train_challenger
from src.data import load_data, split_data
from src.report import charts, write_report
from src.scorecard import Scorecard


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/companies.csv")
    p.add_argument("--out", default="outputs")
    p.add_argument("--max-features", type=int, default=10)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    X_dev, X_test, y_dev, y_test = split_data(load_data(a.data))

    card = Scorecard(max_features=a.max_features).fit(X_dev, y_dev)
    chal = train_challenger(X_dev, y_dev)
    pd_card, pd_chal = card.predict_pd(X_test), chal.predict_proba(X_test)[:, 1]
    dev_scores, test_scores = card.score(X_dev), card.score(X_test)

    s = {"n_dev": len(y_dev), "n_test": len(y_test), "dev_rate": y_dev.mean(), "test_rate": y_test.mean(),
         "card": {**v.discrimination(y_test, pd_card), "brier": brier_score_loss(y_test, pd_card)},
         "chal": {**v.discrimination(y_test, pd_chal), "brier": brier_score_loss(y_test, pd_chal)},
         **v.hosmer_lemeshow(y_test, pd_card), "psi": v.psi(dev_scores, test_scores)}
    s["gini"] = s["card"]["gini"]
    grades = v.rating_grades(dev_scores, test_scores, y_test)
    cal = v.calibration_table(y_test, pd_card)
    found = v.findings(s, grades)
    sh = shap_values(chal, X_test)

    charts(out, y_test, pd_card, pd_chal, cal, dev_scores, test_scores, grades, sh if sh[0] is not None else None)
    card.points_table().to_csv(out / "points_table.csv", index=False)
    write_report(out / "validation_report.md", s, card.coefficients(), grades, cal, found, sh[0] is not None)

    print(f"Scorecard Gini {s['card']['gini']:.3f} | Challenger Gini {s['chal']['gini']:.3f} | PSI {s['psi']:.3f}")
    print("\n".join("- " + f for f in found))
    print(f"\nReport: {out / 'validation_report.md'}")


if __name__ == "__main__":
    main()
