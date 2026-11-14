"""
SkyBurn - Analytics
====================
Derives, from the harmonized time series, the things an emergency manager
actually needs:

  1. Burning calendar  -- the year x month heatmap the challenge asks for
  2. Current status    -- this month against the historical median
  3. Fire season       -- when the season starts and ends
  4. Season shift      -- how far the season has moved over the record
  5. Top months        -- the worst months on record

Everything runs on the harmonized column, so the sensor of origin drops out.
"""

import numpy as np
import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

VALUE_COL = "harmonized"


def _ordinal(n: int) -> str:
    """1 -> 1st, 2 -> 2nd, 11 -> 11th"""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ---------------------------------------------------------------- #
#  1. Burning calendar
# ---------------------------------------------------------------- #

def calendar(series: pd.DataFrame, value_col: str = VALUE_COL) -> dict:
    """
    The year x month grid — this is the "burning activity calendar" the
    challenge statement asks for.
    """
    grid = series.pivot_table(
        index="year", columns="month", values=value_col, aggfunc="sum"
    )
    # make sure all twelve months are present
    grid = grid.reindex(columns=range(1, 13))

    years = [int(y) for y in grid.index]
    values = [[None if pd.isna(v) else float(v) for v in row]
              for row in grid.to_numpy()]

    flat = grid.to_numpy().ravel()
    flat = flat[~np.isnan(flat)]

    return {
        "years": years,
        "months": MONTHS,
        "values": values,
        "max": float(flat.max()) if flat.size else 0.0,
        "p95": float(np.percentile(flat, 95)) if flat.size else 0.0,
    }


# ---------------------------------------------------------------- #
#  2. Current status -- against the historical median
# ---------------------------------------------------------------- #

def current_status(series: pd.DataFrame, value_col: str = VALUE_COL,
                   baseline_years: int = 10, as_of: "pd.Timestamp" = None) -> dict:
    """
    Compare the latest month against the SAME month in previous years.

    We use the median rather than the mean, because a single extreme year drags
    a mean upward. The challenge statement also specifies "historical median".

    PARTIAL MONTHS: if the latest month is still in progress, its counts are not
    comparable to complete months from previous years. Comparing them directly
    is wrong — every current month would look artificially low. So we pro-rate
    by the fraction of the month elapsed, and label the result as a projection.
    """
    if series.empty:
        return {"error": "no data"}

    s = series.sort_values(["year", "month"])
    latest = s.iloc[-1]
    y, m = int(latest["year"]), int(latest["month"])
    observed = float(latest[value_col])
    current = observed

    # is this month still in progress?
    today = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today()
    is_partial = (today.year == y) and (today.month == m)
    elapsed_frac = 1.0

    if is_partial:
        days_in_month = pd.Period(f"{y}-{m:02d}").days_in_month
        elapsed_frac = min(today.day / days_in_month, 1.0)
        if elapsed_frac > 0:
            current = observed / elapsed_frac   # projection for the full month

    # the same month in previous years
    hist = s[(s["month"] == m) & (s["year"] < y)]
    if baseline_years:
        hist = hist[hist["year"] >= y - baseline_years]

    if len(hist) < 3:
        return {
            "year": y, "month": m, "month_name": MONTHS[m - 1],
            "current": round(current),
            "observed": round(observed),
            "is_partial": bool(is_partial),
            "elapsed_pct": round(elapsed_frac * 100),
            "level": "unknown",
            "label": "NO BASELINE",
            "message": f"{MONTHS[m-1]} {y} recorded {round(current):,} hotspots. "
                       f"There is not enough history for this area to compare "
                       f"against a median.",
        }

    vals = hist[value_col].to_numpy()
    median = float(np.median(vals))
    p90 = float(np.percentile(vals, 90))

    ratio = current / median if median > 0 else None
    pct = (current - median) / median * 100 if median > 0 else None

    # Level: above p90 and well above median -> high; moderately above -> elevated
    if current >= p90 and ratio and ratio >= 1.5:
        level = "high"
    elif ratio and ratio >= 1.2:
        level = "elevated"
    elif ratio and ratio <= 0.6:
        level = "low"
    else:
        level = "normal"

    # Rank among the same month across all years.
    # Use the pro-rated projection for a partial month, otherwise the rank and
    # the percentage would contradict each other.
    all_same_month = s[s["month"] == m][["year", value_col]].copy()
    all_same_month.loc[all_same_month["year"] == y, value_col] = current
    all_same_month = all_same_month.sort_values(value_col, ascending=False)
    rank = int((all_same_month["year"] == y).to_numpy().argmax()) + 1

    label = {
        "high": "WELL ABOVE NORMAL",
        "elevated": "ABOVE NORMAL",
        "normal": "NORMAL",
        "low": "BELOW NORMAL",
    }[level]

    if is_partial:
        msg = (
            f"{MONTHS[m-1]} {y} is {today.day} days in, with "
            f"{round(observed):,} hotspots so far, tracking toward about "
            f"{round(current):,} for the full month. The median for "
            f"{MONTHS[m-1]} over the last {len(vals)} years is {round(median):,}"
        )
    else:
        msg = (
            f"{MONTHS[m-1]} {y} recorded {round(current):,} hotspots. "
            f"The median for {MONTHS[m-1]} over the last {len(vals)} years "
            f"is {round(median):,}"
        )
    if pct is not None:
        msg += f" ({pct:+.0f}%)."
    else:
        msg += "."
    msg += f" That ranks {_ordinal(rank)} of {len(all_same_month)} {MONTHS[m-1]}s on record."

    return {
        "year": y, "month": m, "month_name": MONTHS[m - 1],
        "observed": round(observed),
        "current": round(current),
        "is_partial": bool(is_partial),
        "elapsed_pct": round(elapsed_frac * 100),
        "median": round(median),
        "p90": round(p90),
        "ratio": round(ratio, 2) if ratio else None,
        "pct_vs_median": round(pct, 1) if pct is not None else None,
        "level": level,
        "label": label,
        "rank": rank,
        "rank_of": len(all_same_month),
        "baseline_n": len(vals),
        "message": msg,
    }


# ---------------------------------------------------------------- #
#  3. Fire season
# ---------------------------------------------------------------- #

def fire_season(series: pd.DataFrame, value_col: str = VALUE_COL,
                threshold: float = 0.5) -> dict:
    """
    Which months make up the fire season — or seasons, plural.

    A region can have more than one. Pakistani Punjab has two: wheat residue
    burns in April-May and rice residue in October-November, with a quiet
    summer between them. Reporting a single span from the first busy month to
    the last would give "May to November", which is neither true nor useful —
    it hides the gap and buries both peaks.

    So the months above the threshold are grouped into contiguous runs, and
    each run is reported as its own season. December wraps round to January,
    since the calendar is a circle.

    THE THRESHOLD. "At least half the peak month" sounds reasonable and is
    not: where the peaks are sharp, every season collapses to the single peak
    month and the shoulders around it disappear. On Pakistan's record that
    rule returned three one-month seasons — May, November and February — which
    tells a fire manager nothing about when to prepare.

    A month is in season when it is above the year's mean instead. That is the
    ordinary meaning of a busy month, it keeps the shoulders, and it needs no
    arbitrary constant.
    """
    if series.empty:
        return {"error": "no data"}

    monthly = series.groupby("month")[value_col].mean()
    monthly = monthly.reindex(range(1, 13), fill_value=0.0)

    peak_val = float(monthly.max())
    if peak_val <= 0:
        return {"error": "no fire activity on record"}

    peak_month = int(monthly.idxmax())
    cutoff = float(monthly.mean())
    above = [int(m) for m in monthly.index if monthly[m] > cutoff]

    # A flat year has no season at all. Falling back to the peak month here
    # would print "Season: Jan" directly above the words "no distinct season".
    flat = bool(monthly.std() / monthly.mean() < 0.15) if monthly.mean() > 0 else True
    if not above or flat:
        above = []

    seasons = _group_runs(above, monthly)
    season_share = (float(monthly[above].sum() / monthly.sum() * 100)
                    if above else 0.0)

    # Pattern: tightly repeating months suggest agricultural burning;
    # scattered activity suggests wildfire
    cv = float(monthly.std() / monthly.mean()) if monthly.mean() > 0 else 0
    if len(seasons) >= 2:
        pattern = f"{_count_word(len(seasons))} distinct seasons"
        labels = [s["label"] for s in
                  sorted(seasons, key=lambda x: x["months"][0])]
        when = (" and ".join(labels) if len(labels) == 2
                else ", ".join(labels[:-1]) + " and " + labels[-1])
        pattern_note = (
            f"Burning comes in separate bursts — {when} — with quiet months "
            f"between them. Repeating bursts like this are the signature of "
            f"crop-residue burning, where each harvest is followed by a round "
            f"of field clearing, rather than of wildfire."
        )
    elif cv > 1.0:
        pattern = "Sharply seasonal"
        pattern_note = ("Burning is packed into a few months of the year. "
                        "This concentration is typical of scheduled "
                        "agricultural burning rather than wildfire.")
    elif cv > 0.5:
        pattern = "Seasonal"
        pattern_note = ("There is a clear season, but fires are detected "
                        "across the rest of the year as well.")
    else:
        pattern = "Year-round"
        pattern_note = ("Burning is spread through the year with no distinct "
                        "season, which is more typical of wildfire or "
                        "persistent industrial heat sources.")

    return {
        "monthly_mean": [round(float(monthly[m]), 1) for m in range(1, 13)],
        "peak_month": peak_month,
        "peak_month_name": MONTHS[peak_month - 1],
        "seasons": seasons,
        "n_seasons": len(seasons),
        "season_months": above,
        "season_names": [MONTHS[m - 1] for m in above],
        "season_share_pct": round(season_share, 1),
        "pattern": pattern,
        "pattern_note": pattern_note,
        "concentration": round(cv, 2),
    }


def _count_word(n: int) -> str:
    return {2: "Two", 3: "Three", 4: "Four"}.get(n, str(n))


def _group_runs(months: list, monthly) -> list:
    """
    Group months into contiguous runs, wrapping December round to January.

    [4, 5, 10, 11]  ->  two runs: Apr-May and Oct-Nov
    [11, 12, 1]     ->  one run spanning the year end
    """
    if not months:
        return []

    present = set(months)
    if len(present) == 12:
        return [_describe_run(list(range(1, 13)), monthly)]

    # start at a month whose predecessor is absent, so no run is split
    start = next(m for m in sorted(present) if ((m - 2) % 12) + 1 not in present)

    runs, current = [], []
    for i in range(12):
        m = ((start - 1 + i) % 12) + 1
        if m in present:
            current.append(m)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    out = [_describe_run(r, monthly) for r in runs]
    out.sort(key=lambda s: s["total"], reverse=True)
    return out


def _describe_run(run: list, monthly) -> dict:
    total = float(monthly[run].sum())
    peak = max(run, key=lambda m: monthly[m])
    return {
        "months": run,
        "names": [MONTHS[m - 1] for m in run],
        "label": (MONTHS[run[0] - 1] if len(run) == 1
                  else f"{MONTHS[run[0] - 1]}–{MONTHS[run[-1] - 1]}"),
        "peak_month": int(peak),
        "peak_month_name": MONTHS[peak - 1],
        "total": round(total, 1),
    }


# ---------------------------------------------------------------- #
#  4. Season shift -- climate signal
# ---------------------------------------------------------------- #

def season_shift(series: pd.DataFrame, value_col: str = VALUE_COL) -> dict:
    """
    Finds each year's "centre of mass" — the point in the year by which half
    the burning has happened — then tests whether that point is drifting
    earlier or later across the record.

    This is rarely shown in fire dashboards and is the measure that carries
    real climate meaning.

    NOTE: months are circular data (January follows December), so this uses a
    circular mean rather than a plain average.
    """
    if series.empty:
        return {"error": "no data"}

    rows = []
    for year, grp in series.groupby("year"):
        total = grp[value_col].sum()
        if total <= 0:
            continue
        # circular mean: convert the month to an angle
        ang = (grp["month"].to_numpy() - 1) * (2 * np.pi / 12)
        w = grp[value_col].to_numpy()
        x = float((w * np.cos(ang)).sum() / total)
        y = float((w * np.sin(ang)).sum() / total)
        if x == 0 and y == 0:
            continue
        mean_ang = np.arctan2(y, x) % (2 * np.pi)
        center = mean_ang / (2 * np.pi) * 12 + 1   # 1..13
        rows.append({"year": int(year), "center_month": round(float(center), 2),
                     "total": float(total)})

    if len(rows) < 8:
        return {"error": "not enough years to establish a trend"}

    df = pd.DataFrame(rows)

    # Trend on circular data: unwrap first so the Dec->Jan boundary does not
    # create a false jump
    c = df["center_month"].to_numpy()
    c_unwrapped = np.unwrap(c * (2 * np.pi / 12)) * (12 / (2 * np.pi))

    yrs = df["year"].to_numpy().astype(float)
    slope, intercept = np.polyfit(yrs, c_unwrapped, 1)
    per_decade = slope * 10

    # how strong the trend is
    r = float(np.corrcoef(yrs, c_unwrapped)[0, 1])

    span = int(yrs.max() - yrs.min())
    if abs(per_decade) < 0.15 or abs(r) < 0.3:
        verdict = ("The timing of the fire season has held steady over this "
                   "record, with no clear drift through the calendar.")
        direction = "stable"
    else:
        days = abs(per_decade) * 30.4
        total = days * span / 10
        if per_decade < 0:
            direction = "earlier"
            verdict = (f"The fire season is arriving about {days:.0f} days "
                       f"earlier each decade — roughly {total:.0f} days "
                       f"earlier across the {span}-year record.")
        else:
            direction = "later"
            verdict = (f"The fire season is arriving about {days:.0f} days "
                       f"later each decade — roughly {total:.0f} days "
                       f"later across the {span}-year record.")

    return {
        "by_year": rows,
        "shift_months_per_decade": round(float(per_decade), 2),
        "shift_days_per_decade": round(float(per_decade) * 30.4, 1),
        "direction": direction,
        "correlation": round(r, 2),
        "verdict": verdict,
        "years_covered": [int(yrs.min()), int(yrs.max())],
    }


# ---------------------------------------------------------------- #
#  5. Top months
# ---------------------------------------------------------------- #

def top_months(series: pd.DataFrame, n: int = 10,
               value_col: str = VALUE_COL) -> list:
    """The worst months on record."""
    s = series.nlargest(n, value_col)
    return [
        {
            "year": int(r["year"]),
            "month": int(r["month"]),
            "label": f"{MONTHS[int(r['month'])-1]} {int(r['year'])}",
            "value": round(float(r[value_col])),
        }
        for _, r in s.iterrows()
    ]


# ---------------------------------------------------------------- #

def full_report(series: pd.DataFrame, meta: dict = None,
                check: dict = None, value_col: str = VALUE_COL) -> dict:
    """Everything in one dict — this is what the API serves to the frontend."""
    return {
        "calendar": calendar(series, value_col),
        "status": current_status(series, value_col),
        "season": fire_season(series, value_col),
        "shift": season_shift(series, value_col),
        "top_months": top_months(series, 10, value_col),
        "harmonization": {
            "meta": meta or {},
            "check": check or {},
        },
        "coverage": {
            "start_year": int(series["year"].min()) if not series.empty else None,
            "end_year": int(series["year"].max()) if not series.empty else None,
            "months": len(series),
            "total_hotspots": int(series[value_col].sum()) if not series.empty else 0,
        },
    }


if __name__ == "__main__":
    import json

    series = pd.read_csv("data/series.csv")
    rep = full_report(series)

    print("=" * 60)
    print("CURRENT STATUS")
    print("=" * 60)
    print(" ", rep["status"]["message"])
    print("  Level:", rep["status"]["label"])

    print("\n" + "=" * 60)
    print("FIRE SEASON")
    print("=" * 60)
    s = rep["season"]
    for i, run in enumerate(s.get("seasons") or [], 1):
        tag = f"Season {i}" if len(s["seasons"]) > 1 else "Season"
        print(f"  {tag}: {run['label']}  (peak {run['peak_month_name']})")
    print(f"  Share of annual burning in season: {s['season_share_pct']}%")
    print(f"  Pattern: {s['pattern']} -- {s['pattern_note']}")

    print("\n" + "=" * 60)
    print("SEASON SHIFT")
    print("=" * 60)
    sh = rep["shift"]
    if "error" in sh:
        print(" ", sh["error"])
    else:
        print(" ", sh["verdict"])
        print(f"  Correlation: {sh['correlation']}  ({sh['years_covered'][0]}-{sh['years_covered'][1]})")

    print("\n" + "=" * 60)
    print("WORST MONTHS ON RECORD")
    print("=" * 60)
    for i, t in enumerate(rep["top_months"][:5], 1):
        print(f"  {i}. {t['label']}: {t['value']:,}")

    with open("data/report.json", "w") as f:
        json.dump(rep, f, indent=2)
    print("\n-> data/report.json")