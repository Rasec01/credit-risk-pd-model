"""Write the validation report (Markdown) and its charts."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def charts(out: Path, y_test, pd_card, pd_chal, cal_card, dev_scores, test_scores, grades, shap=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, p in [("Scorecard", pd_card), ("Challenger", pd_chal)]:
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test sample)")
    ax.legend()
    _save(fig, out / "roc.png")

    fig, ax = plt.subplots(figsize=(5, 5))
    m = max(cal_card["predicted"].max(), cal_card["observed"].max()) * 1.1
    ax.plot([0, m], [0, m], "--", color="grey")
    ax.plot(cal_card["predicted"], cal_card["observed"], "o-")
    ax.set(xlabel="Predicted PD", ylabel="Observed default rate", title="Calibration, scorecard (test)")
    _save(fig, out / "calibration.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(dev_scores.min(), test_scores.min()), max(dev_scores.max(), test_scores.max()), 30)
    ax.hist(dev_scores, bins, alpha=0.5, density=True, label="Development")
    ax.hist(test_scores, bins, alpha=0.5, density=True, label="Test")
    ax.set(xlabel="Score (higher = safer)", title="Score distribution")
    ax.legend()
    _save(fig, out / "score_distribution.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(grades.index.astype(str), grades["default_rate"] * 100)
    ax.set(xlabel="Grade (1 = safest)", ylabel="Default rate, %", title="Default rate by grade (test)")
    _save(fig, out / "grades.png")

    if shap is not None:
        import shap as shap_lib
        values, sample = shap
        shap_lib.summary_plot(values, sample, show=False, max_display=12)
        plt.gcf().set_size_inches(8, 6)
        _save(plt.gcf(), out / "shap_challenger.png")


def _table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    rows = ["| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return head + "\n".join(rows) + "\n"


def write_report(path: Path, s: dict, coefs, grades, cal, findings, has_shap: bool):
    metrics = pd.DataFrame({
        "Scorecard": [s["card"]["auc"], s["card"]["gini"], s["card"]["ks"], s["card"]["brier"]],
        "Challenger": [s["chal"]["auc"], s["chal"]["gini"], s["chal"]["ks"], s["chal"]["brier"]],
    }, index=pd.Index(["AUC", "Gini", "KS", "Brier score"], name="metric"))
    text = f"""# PD model validation report

Development sample: {s['n_dev']} firms ({s['dev_rate']:.2%} defaulted). Test sample: {s['n_test']} firms ({s['test_rate']:.2%} defaulted).
All figures below are on the test sample unless stated.

## Findings

{chr(10).join('- ' + f for f in findings)}
- Challenger vs scorecard: Gini {s['chal']['gini']:.2f} vs {s['card']['gini']:.2f}. {"The challenger ranks firms better; weigh that against the scorecard's transparency." if s['chal']['gini'] > s['card']['gini'] + 0.02 else "The challenger does not clearly beat the scorecard, so the transparent model is preferred."}

## Discrimination

{_table(metrics)}
![ROC](roc.png)

## Scorecard variables

Selected by information value (IV), dropping ratios correlated above 0.7 with a stronger one.
Coefficients should be negative: a higher weight of evidence means a safer firm.

{_table(coefs.set_index('feature'))}
The full points table is in `points_table.csv`.

## Calibration (scorecard)

Hosmer-Lemeshow statistic {s['hl_statistic']:.2f}, p-value {s['hl_p_value']:.3f}.

{_table(cal)}
![Calibration](calibration.png)

## Stability

Population stability index of the score, development vs test: **{s['psi']:.3f}**.

![Score distribution](score_distribution.png)

## Rating grades

{_table(grades, "{:.4f}")}
![Grades](grades.png)

{"## Challenger explanation" + chr(10) + chr(10) + "![SHAP](shap_challenger.png)" + chr(10) if has_shap else ""}
## Limitations

- The public data has no dates, so the test sample is a random hold-out, not a later period. A bank would also test out of time.
- Default here means a bankruptcy filing, not the 90-days-past-due definition used under Basel.
- The sample's default rate need not match a lending portfolio's; PDs would be recalibrated to the portfolio's long-run default rate before use.
"""
    path.write_text(text)
