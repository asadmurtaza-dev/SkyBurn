import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRUTH = Path("data/truth.csv")
SERIES = Path("data/series.csv")


def bias_report(df: pd.DataFrame, col: str, label: str) -> dict:
    """Mean error of a column against the truth, overall and by era."""
    ratio = df[col] / df["truth"]
    out = {"label": label, "overall": float(ratio.mean())}

    eras = [
        ("2001-2011  MODIS only",      (2001, 2011)),
        ("2012-2017  S-NPP",           (2012, 2017)),
        ("2018-2026  S-NPP + NOAA-20", (2018, 2026)),
    ]
    for name, (lo, hi) in eras:
        m = df[(df["year"] >= lo) & (df["year"] <= hi)]
        if m.empty:
            continue
        out[name] = float((m[col] / m["truth"]).mean())
    return out


def step_at(df: pd.DataFrame, col: str, year: int, window: int = 3) -> float:
    """Percentage step across a transition year."""
    ann = df.groupby("year")[col].sum()
    before = ann.loc[year - window:year - 1]
    after = ann.loc[year + 1:year + window]
    if before.empty or after.empty or before.mean() <= 0:
        return float("nan")
    return (after.mean() - before.mean()) / before.mean() * 100


def main():
    if not TRUTH.exists() or not SERIES.exists():
        print("Missing data. Run these first:")
        print("    python3 make_demo_data.py")
        print("    python3 harmonize.py")
        return 1

    truth = pd.read_csv(TRUTH)
    series = pd.read_csv(SERIES)
    df = series.merge(truth, on=["year", "month"], how="inner")
    df = df[df["truth"] > 0]

    if df.empty:
        print("No overlapping months between the series and the truth.")
        return 1

    print("=" * 66)
    print("SkyBurn - validation against known truth")
    print("=" * 66)
    print(f"  {len(df)} months, {int(df['year'].min())}-{int(df['year'].max())}")

    # ---- 1. bias by era ----
    print("\n" + "-" * 66)
    print("1. BIAS  (harmonized / truth, 1.00 is perfect)")
    print("-" * 66)

    harm = bias_report(df, "harmonized", "harmonized")
    raw = bias_report(df, "raw", "raw")

    print(f"  {'':30s} {'harmonized':>12s} {'raw':>12s}")
    for key in harm:
        if key == "label":
            continue
        print(f"  {key:30s} {harm[key]:>12.3f} {raw.get(key, float('nan')):>12.3f}")

    # ---- 2. steps at the transitions ----
    print("\n" + "-" * 66)
    print("2. STEPS AT SENSOR TRANSITIONS")
    print("-" * 66)
    print("  Compared against the step the TRUTH actually takes. A short")
    print("  window is noisy, so what matters is the distance from truth,")
    print("  not the distance from zero.\n")

    transitions = [
        (2012, "MODIS -> S-NPP"),
        (2018, "NOAA-20 joins S-NPP (double-count test)"),
    ]
    print(f"  {'':44s} {'truth':>8s} {'harmon.':>9s} {'raw':>9s}")
    errors = {}
    for year, name in transitions:
        t = step_at(df, "truth", year)
        h = step_at(df, "harmonized", year)
        r = step_at(df, "raw", year)
        errors[year] = abs(h - t)
        print(f"  {year}  {name:38s} {t:>+7.1f}% {h:>+8.1f}% {r:>+8.1f}%")
    print()
    for year in errors:
        print(f"  {year}: harmonized is {errors[year]:.1f} points from the truth's step")

    # ---- 3. correlation with the truth ----
    print("\n" + "-" * 66)
    print("3. AGREEMENT WITH TRUTH")
    print("-" * 66)
    for col in ("harmonized", "raw"):
        r = float(np.corrcoef(df[col], df["truth"])[0, 1])
        mape = float((abs(df[col] - df["truth"]) / df["truth"]).mean() * 100)
        print(f"  {col:12s}  r = {r:.3f}    mean abs error = {mape:5.1f}%")

    # ---- verdict ----
    print("\n" + "=" * 66)
    bias_err = abs(harm["overall"] - 1.0) * 100
    era_errs = [abs(v - 1.0) * 100 for k, v in harm.items()
                if k not in ("label", "overall")]
    worst_era = max(era_errs) if era_errs else 0.0

    checks = [
        ("overall bias within 5%", bias_err < 5, f"{bias_err:.1f}% off"),
        ("every era within 10%", worst_era < 10, f"worst era {worst_era:.1f}% off"),
        ("2012 step within 12pts of truth", errors[2012] < 12,
         f"{errors[2012]:.1f} pts"),
        ("2018 step within 12pts of truth", errors[2018] < 12,
         f"{errors[2018]:.1f} pts"),
    ]
    passed = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:28s} {detail}")

    print()
    if passed == len(checks):
        print("  All checks passed — the harmonization recovers the known answer.")
        return 0
    print(f"  {passed}/{len(checks)} checks passed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())