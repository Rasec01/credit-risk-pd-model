"""
Download the Taiwanese company-bankruptcy data (UCI repository, id 572): 6,819
firms, 95 financial ratios, 1999-2009. Saves data/companies.csv with the target
column named 'default'.
"""
from pathlib import Path

from ucimlrepo import fetch_ucirepo

data = fetch_ucirepo(id=572).data
df = data.features.copy()
df["default"] = data.targets.iloc[:, 0].astype(int)

out = Path(__file__).resolve().parents[1] / "data" / "companies.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {len(df)} firms to {out} ({df['default'].mean():.1%} defaulted)")
