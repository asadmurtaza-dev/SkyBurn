"""
SkyBurn - Harmonization engine
================================
The core problem this challenge poses: the hotspot record is split across
satellites whose counts cannot be compared directly.

Why not:
  - a MODIS hotspot is the centre of a 1 km pixel
  - a VIIRS hotspot is the centre of a 375 m pixel
  - VIIRS therefore detects smaller fires, and performs better at night
  => VIIRS reports far more hotspots than MODIS for THE SAME fires.

And it is not only MODIS vs VIIRS. There are several VIIRS satellites — Suomi
NPP (2012-2026), NOAA-20 (2018-) and NOAA-21 (2023-) — which flew at the same
time. Adding them together counts the same fire two or three times, and creates
a fresh artificial step every time a new satellite launches.

So the record is treated as a set of PLATFORMS, and every platform is placed on
the scale of one reference platform:

  1. Pick a reference platform (the one overlapping the most of the record)
  2. For every other platform, measure its ratio to the reference over the
     period where both were flying
  3. For each month, use exactly ONE platform — the best one available — and
     rescale it to the reference
  => one continuous series, no double counting, no launch-day steps

This is an approximation, not an official product. Say so when presenting it.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- #
#  Config
# ---------------------------------------------------------------- #

GRID_DEG = 0.25          # cell size in degrees (~27 km)
FALLBACK_RATIO = 2.5     # MODIS -> VIIRS default when no overlap exists

# How strongly a cell's own ratio is pulled toward the global one, expressed
# in platform counts. A cell with this many detections sits halfway between
# its own estimate and the global ratio. Higher = smoother, more global.
SHRINKAGE_COUNTS = 25

# Which platform to express everything in. Suomi NPP is the natural choice:
# it overlaps MODIS at one end and the NOAA satellites at the other, so every
# other platform can be tied to it directly.
REFERENCE_PREFERENCE = [
    "VIIRS_SNPP",
    "VIIRS_NOAA20",
    "VIIRS_NOAA21",
    "MODIS",
]

VIIRS_PLATFORMS = {"VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21"}


# ---------------------------------------------------------------- #
#  Step 1: grid + monthly counts per platform
# ---------------------------------------------------------------- #

def add_grid(df: pd.DataFrame, cell: float = GRID_DEG) -> pd.DataFrame:
    """Assign every hotspot to a grid cell."""
    out = df.copy()
    out["cell_lat"] = (np.floor(out["lat"] / cell) * cell).round(4)
    out["cell_lon"] = (np.floor(out["lon"] / cell) * cell).round(4)
    out["cell"] = out["cell_lat"].astype(str) + "_" + out["cell_lon"].astype(str)
    return out


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hotspots -> a count per (cell, platform, year, month).
    Everything downstream is built on this table.
    """
    d = add_grid(df)
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month

    if "platform" not in d.columns:
        # older callers passed only a sensor family
        d["platform"] = d.get("sensor", "MODIS")

    counts = (
        d.groupby(["cell", "cell_lat", "cell_lon", "platform", "year", "month"])
        .size()
        .reset_index(name="count")
    )
    return counts


MIN_MONTH_COVERAGE = 0.8   # a boundary month must be this complete to count


def platform_spans(counts: pd.DataFrame, date_ranges: dict = None) -> dict:
    """
    The operating window of each platform, as (first, last) year-month ordinals.

    A platform is only ever used inside its own window — outside it, zero
    detections mean "not flying", not "no fires".

    PARTIAL BOUNDARY MONTHS. A platform rarely starts or stops on the first of
    a month. Suomi NPP began on 20 January 2012 and stopped on 2 November 2026,
    so each of those months holds only a few days of observations. Treating
    them as whole months makes the record look like burning collapsed — and
    since the last month is what the "current conditions" panel reports, that
    error lands in the most prominent number on the page.

    A STOPPED platform is not the same as a CURRENT one. The record simply
    ends partway through the present month for every platform still flying —
    that is not a defect, and the analytics pro-rate it. Only a platform whose
    data stops well before the rest of the record has genuinely ended, and only
    that platform's final month gets dropped.

    `date_ranges` maps platform -> (first_date, last_date); without it the
    trimming is skipped.
    """
    c = counts[counts["count"] > 0].copy()
    c["ym"] = c["year"] * 12 + (c["month"] - 1)

    # How current the dataset is overall. A platform reaching this date is
    # still flying; one stopping well short of it has ended.
    data_end = None
    if date_ranges:
        ends = [v[1] for v in date_ranges.values() if pd.notna(v[1])]
        if ends:
            data_end = max(ends)

    spans = {}
    for platform, grp in c.groupby("platform"):
        lo, hi = int(grp["ym"].min()), int(grp["ym"].max())

        if date_ranges and platform in date_ranges:
            first, last = date_ranges[platform]

            # First month: a platform that began mid-month really does have a
            # short first month, so drop it when it is mostly missing.
            if pd.notna(first):
                days = pd.Period(f"{first.year}-{first.month:02d}").days_in_month
                covered = (days - first.day + 1) / days
                if covered < MIN_MONTH_COVERAGE:
                    lo += 1

            # Last month: only trim if this platform STOPPED. If it runs to
            # within a week of the newest data anywhere, it is still flying and
            # its short final month is just the present month.
            if pd.notna(last) and data_end is not None:
                stopped = (data_end - last).days > 7
                days = pd.Period(f"{last.year}-{last.month:02d}").days_in_month
                covered = last.day / days
                if stopped and covered < MIN_MONTH_COVERAGE:
                    print(f"  [!] {platform} ended {last.date()}, "
                          f"dropping its partial final month")
                    hi -= 1

        if lo <= hi:
            spans[platform] = (lo, hi)
        else:
            print(f"  [!] {platform}: no complete months, skipped")

    return spans


def platform_date_ranges(hotspots: pd.DataFrame) -> dict:
    """First and last observation date for each platform."""
    if "platform" not in hotspots.columns:
        return {}
    g = hotspots.groupby("platform")["date"]
    return {p: (g.min()[p], g.max()[p]) for p in g.min().index}


def pick_reference(spans: dict) -> str:
    """Choose the reference platform: preference order, then longest record."""
    available = [p for p in REFERENCE_PREFERENCE if p in spans]
    if available:
        return available[0]
    if not spans:
        return ""
    return max(spans, key=lambda p: spans[p][1] - spans[p][0])


# ---------------------------------------------------------------- #
#  Step 2: calibrate every platform against the reference
# ---------------------------------------------------------------- #

def compute_ratios(counts: pd.DataFrame, reference: str, spans: dict):
    """
    For each platform, the factor that converts its counts to reference counts.

    ratio = reference_total / platform_total, measured only over the months
    where BOTH were flying. Multiplying a platform's count by its ratio puts it
    on the reference scale.

    IMPORTANT (this mistake is easy to make):
    Do not filter cell-months on "at least N detections in both" before taking
    the ratio. That keeps only the months where the weaker sensor happened to
    score well, which biases the ratio downward — selection bias. Total the
    whole overlap period instead, and apply thresholds to a cell's TOTAL.

    Returns:
        global_ratios : {platform: float}
        cell_ratios   : {platform: DataFrame[cell, ratio]}
        overlaps      : {platform: (first_year, last_year, n_months)}
    """
    wide = counts.pivot_table(
        index=["cell", "year", "month"],
        columns="platform",
        values="count",
        fill_value=0,
    ).reset_index()

    for p in spans:
        if p not in wide.columns:
            wide[p] = 0

    wide["ym"] = wide["year"] * 12 + (wide["month"] - 1)

    global_ratios = {reference: 1.0}
    cell_ratios = {}
    overlaps = {}

    ref_lo, ref_hi = spans[reference]

    for platform in spans:
        if platform == reference:
            continue

        lo, hi = spans[platform]
        ov_lo, ov_hi = max(lo, ref_lo), min(hi, ref_hi)

        if ov_lo > ov_hi:
            # never flew at the same time — fall back on the sensor families
            ratio = _family_fallback(platform, reference)
            global_ratios[platform] = ratio
            overlaps[platform] = (None, None, 0)
            print(f"  [!] {platform}: no overlap with {reference}, "
                  f"using fallback ratio {ratio}")
            continue

        ov = wide[(wide["ym"] >= ov_lo) & (wide["ym"] <= ov_hi)]

        total_ref = ov[reference].sum()
        total_plat = ov[platform].sum()

        if total_plat <= 0 or total_ref <= 0:
            ratio = _family_fallback(platform, reference)
            global_ratios[platform] = ratio
            overlaps[platform] = (None, None, 0)
            print(f"  [!] {platform}: empty overlap, using fallback {ratio}")
            continue

        global_ratios[platform] = float(total_ref / total_plat)
        overlaps[platform] = (ov_lo // 12, ov_hi // 12, int(ov_hi - ov_lo + 1))

        # Per-cell ratios, because the detection advantage differs between
        # dense forest and open cropland.
        per_cell = (
            ov.groupby("cell")
            .agg(ref=(reference, "sum"), plat=(platform, "sum"),
                 n_months=("month", "count"))
            .reset_index()
        )

        if not per_cell.empty:
            # SELECTION BIAS, AGAIN — this time at cell level.
            #
            # The obvious move is to keep only cells with enough counts, e.g.
            # plat >= 20. But `plat` is the ratio's denominator, so that keeps
            # the cells where this platform happened to over-detect, and the
            # ratios come out systematically low. Measured on synthetic data
            # with a known 2.6 ratio, a plat >= 20 filter returned 2.41 — a
            # 10% error that reappears as a step at the era boundary.
            #
            # So: no threshold, no selection. Every cell is kept and shrunk
            # toward the global ratio in proportion to how little data it has.
            # A cell with no counts lands exactly on the global ratio; a cell
            # with plenty keeps its own.
            w = SHRINKAGE_COUNTS
            g = global_ratios[platform]
            per_cell["ratio"] = (per_cell["ref"] + w * g) / (per_cell["plat"] + w)

            # Renormalise so the per-cell ratios reproduce the overall ratio
            # exactly. Without this the per-cell path and the global path
            # disagree, which shows up as a step at the era boundary — the
            # very artefact this whole module exists to remove.
            rescaled_total = (per_cell["plat"] * per_cell["ratio"]).sum()
            covered_ref = per_cell["ref"].sum()
            if rescaled_total > 0 and covered_ref > 0:
                per_cell["ratio"] *= covered_ref / rescaled_total

            cell_ratios[platform] = per_cell[["cell", "ratio"]]

    return global_ratios, cell_ratios, overlaps


def _family_fallback(platform: str, reference: str) -> float:
    """A rough ratio when two platforms never flew together."""
    p_viirs = platform in VIIRS_PLATFORMS
    r_viirs = reference in VIIRS_PLATFORMS
    if p_viirs == r_viirs:
        return 1.0
    return FALLBACK_RATIO if r_viirs else 1.0 / FALLBACK_RATIO


# ---------------------------------------------------------------- #
#  Step 3: one platform per month, rescaled to the reference
# ---------------------------------------------------------------- #

def harmonize(counts: pd.DataFrame, date_ranges: dict = None) -> tuple:
    """
    Produce one harmonized count per cell-month.

    For each month, exactly ONE platform is used — never a sum. The platform is
    chosen by priority: the reference where it was flying, otherwise the
    best-calibrated platform that was. Its count is then multiplied by that
    platform's ratio to put it on the reference scale.

    There is a subtle trap here. Within the reference's own era, a cell-month
    where the reference saw nothing but another platform saw something must
    still read zero. The reference is the measurement; if it saw nothing, the
    answer for that month is nothing. Hence the decision is made by era, not by
    whichever platform happens to be non-zero.
    """
    spans = platform_spans(counts, date_ranges)
    if not spans:
        raise ValueError("No platforms with any detections.")

    reference = pick_reference(spans)
    global_ratios, cell_ratios, overlaps = compute_ratios(counts, reference, spans)

    wide = counts.pivot_table(
        index=["cell", "cell_lat", "cell_lon", "year", "month"],
        columns="platform",
        values="count",
        fill_value=0,
    ).reset_index()

    platforms = list(spans.keys())
    for p in platforms:
        if p not in wide.columns:
            wide[p] = 0

    wide["ym"] = wide["year"] * 12 + (wide["month"] - 1)

    # Priority when the reference is unavailable: the platform needing the
    # LEAST rescaling, because every unit of rescaling is a unit of
    # assumption. A sister VIIRS satellite sits near 1.0 and is close to a
    # drop-in substitute; MODIS needs a ~2.7x scale-up and is a last resort,
    # however long its record. Ties go to the longer overlap.
    #
    # Ranking by overlap length instead would pick MODIS over NOAA-20 for the
    # present month purely because MODIS has flown longer — exactly backwards.
    others = [p for p in platforms if p != reference]
    others.sort(key=lambda p: (
        abs(np.log(max(global_ratios.get(p, 1.0), 1e-9))),
        -overlaps.get(p, (None, None, 0))[2],
    ))
    priority = [reference] + others

    # raw = the naive stack of everything, kept so the app can show the error
    wide["raw"] = wide[platforms].sum(axis=1)

    harmonized = np.zeros(len(wide))
    chosen = np.full(len(wide), "", dtype=object)
    filled = np.zeros(len(wide), dtype=bool)

    for platform in priority:
        lo, hi = spans[platform]
        operating = ((wide["ym"] >= lo) & (wide["ym"] <= hi)).to_numpy()
        take = operating & ~filled
        if not take.any():
            continue

        # per-cell ratio where available, else the platform's global ratio
        ratio = pd.Series(global_ratios[platform], index=wide.index)
        if platform in cell_ratios:
            local = wide[["cell"]].merge(cell_ratios[platform], on="cell", how="left")
            ratio = local["ratio"].fillna(global_ratios[platform])

        counts_here = wide.loc[take, platform].to_numpy()
        values = counts_here * ratio.to_numpy()[take]

        # Per-cell ratios decide WHERE the activity sits; the global ratio
        # decides HOW MUCH there was. Without this the two disagree.
        #
        # A per-cell ratio is a ratio of two noisy counts, and the expected
        # value of such a ratio sits above the true one (Jensen's inequality,
        # since 1/x is convex). Multiply independent counts by those ratios and
        # the reconstructed era comes out systematically high — measured on
        # synthetic data with a known answer, the MODIS era landed 17% above
        # the truth. Rescaling the era total back onto the globally calibrated
        # total removes that, while the per-cell ratios still redistribute the
        # activity between cells.
        if platform != reference:
            target = counts_here.sum() * global_ratios[platform]
            if values.sum() > 0 and target > 0:
                values = values * (target / values.sum())

        harmonized[take] = values
        chosen[take] = platform
        filled |= take

    wide["harmonized"] = np.round(harmonized)
    wide["source_platform"] = chosen
    wide["method"] = np.where(
        wide["source_platform"] == reference, "reference", "rescaled"
    )

    # Family totals, for display only — these are raw, uncalibrated sums and
    # must not be used for analysis.
    viirs_cols = [p for p in platforms if p in VIIRS_PLATFORMS]
    modis_cols = [p for p in platforms if p not in VIIRS_PLATFORMS]
    wide["VIIRS"] = wide[viirs_cols].sum(axis=1) if viirs_cols else 0
    wide["MODIS"] = wide[modis_cols].sum(axis=1) if modis_cols else 0

    meta = {
        "reference": reference,
        "platforms": platforms,
        "global_ratios": {p: round(r, 3) for p, r in global_ratios.items()},
        "overlaps": {p: list(v) for p, v in overlaps.items()},
        "cells_calibrated": {p: len(df) for p, df in cell_ratios.items()},
        "cells_total": wide["cell"].nunique(),
        "spans": {p: [v[0] // 12, v[1] // 12] for p, v in spans.items()},
    }
    # kept for backwards compatibility with the app's display
    non_ref_viirs = [p for p in platforms if p in VIIRS_PLATFORMS and p != reference]
    if "MODIS" in global_ratios and reference in VIIRS_PLATFORMS:
        meta["global_ratio"] = round(global_ratios["MODIS"], 3)
    elif non_ref_viirs:
        meta["global_ratio"] = round(global_ratios[non_ref_viirs[0]], 3)
    else:
        meta["global_ratio"] = 1.0

    return wide, meta


# ---------------------------------------------------------------- #
#  Step 4: a time series for one area
# ---------------------------------------------------------------- #

def area_series(harmonized: pd.DataFrame, bbox: tuple = None) -> pd.DataFrame:
    """
    Collapse the grid cells inside a bounding box into one monthly series.
    bbox = (west, south, east, north), or None for everything.
    """
    d = harmonized
    if bbox:
        w, s, e, n = bbox
        d = d[
            (d["cell_lon"] >= w) & (d["cell_lon"] <= e)
            & (d["cell_lat"] >= s) & (d["cell_lat"] <= n)
        ]

    agg = {
        "harmonized": ("harmonized", "sum"),
        "raw": ("raw", "sum"),
    }
    for col in ("MODIS", "VIIRS"):
        if col in d.columns:
            agg[col.lower()] = (col, "sum")

    series = (
        d.groupby(["year", "month"])
        .agg(**agg)
        .reset_index()
        .sort_values(["year", "month"])
    )

    for col in ("modis", "viirs"):
        if col not in series.columns:
            series[col] = 0

    series["date"] = pd.to_datetime(
        dict(year=series["year"], month=series["month"], day=1)
    )
    return series.reset_index(drop=True)


# ---------------------------------------------------------------- #
#  Step 5: discontinuity test -- did the harmonization work?
# ---------------------------------------------------------------- #

def discontinuity_check(series: pd.DataFrame, switch_year: int = None,
                        meta: dict = None) -> dict:
    """
    The number to put in front of a reviewer.

    Measures the step across the main sensor transition in the raw record and
    in the harmonized record. If the harmonization is doing its job, the
    harmonized step should be substantially smaller.
    """
    if switch_year is None:
        switch_year = _main_transition(series, meta)
    if switch_year is None:
        return {"error": "could not locate a sensor transition"}

    before = series[(series["year"] >= switch_year - 3) & (series["year"] < switch_year)]
    after = series[(series["year"] > switch_year) & (series["year"] <= switch_year + 3)]

    if before.empty or after.empty:
        return {"error": f"not enough data around {switch_year}"}

    def jump(col):
        b, a = before[col].mean(), after[col].mean()
        if b <= 0:
            return None
        return round((a - b) / b * 100, 1)

    raw_jump = jump("raw")
    harm_jump = jump("harmonized")

    result = {
        "switch_year": int(switch_year),
        "raw_jump_pct": raw_jump,
        "harmonized_jump_pct": harm_jump,
    }

    if raw_jump is not None and harm_jump is not None:
        improvement = abs(raw_jump) - abs(harm_jump)
        result["improvement_pts"] = round(improvement, 1)
        result["verdict"] = (
            f"Across the {switch_year} sensor transition the raw record jumps "
            f"{raw_jump:+.0f}%, while the harmonized record moves {harm_jump:+.0f}% "
            f"— a {improvement:.0f}-point reduction in the artificial step."
        )

    return result


def _main_transition(series: pd.DataFrame, meta: dict = None) -> int:
    """The year the reference platform began — the biggest step in the record."""
    if meta and meta.get("spans") and meta.get("reference"):
        span = meta["spans"].get(meta["reference"])
        if span:
            return int(span[0])
    if series.empty:
        return None
    return int(series["year"].min()) + 11


# ---------------------------------------------------------------- #

def run(hotspots: pd.DataFrame, bbox: tuple = None, switch_year: int = None):
    """The whole pipeline in one call."""
    print("Building monthly counts ...")
    counts = monthly_counts(hotspots)
    date_ranges = platform_date_ranges(hotspots)
    print(f"  {len(counts):,} cell-month records, {counts['cell'].nunique():,} cells")
    for p, n in counts.groupby("platform")["count"].sum().items():
        rng = date_ranges.get(p)
        when = f"{rng[0].date()} .. {rng[1].date()}" if rng else ""
        print(f"    {p:14s} {int(n):>10,} hotspots   {when}")

    print("Calibrating platforms ...")
    harm, meta = harmonize(counts, date_ranges)
    print(f"  reference: {meta['reference']}")
    for p, r in meta["global_ratios"].items():
        if p == meta["reference"]:
            continue
        ov = meta["overlaps"].get(p, [None, None, 0])
        where = f"{ov[0]}-{ov[1]}, {ov[2]} months" if ov[2] else "no overlap"
        print(f"    {p:14s} x{r:<7.3f} ({where})")

    print("Building the time series ...")
    series = area_series(harm, bbox)
    print(f"  {len(series)} months, {series['year'].min()}-{series['year'].max()}")

    check = discontinuity_check(series, switch_year, meta)
    if "verdict" in check:
        print(f"\n  {check['verdict']}")
    elif "error" in check:
        print(f"\n  [!] {check['error']}")

    return {"counts": counts, "harmonized": harm, "series": series,
            "meta": meta, "check": check}


if __name__ == "__main__":
    import json
    from firms_fetch import load_archive, normalise

    print("Loading the archive ...")
    raw = load_archive()
    clean = normalise(raw)
    print(f"  {len(clean):,} hotspots\n")

    result = run(clean)

    result["series"].to_csv("data/series.csv", index=False)
    result["harmonized"].to_csv("data/harmonized_grid.csv", index=False)

    # Save the metadata for the app to read. The app must NOT recompute the
    # ratios itself — doing that from the grid reintroduces the selection bias
    # described in compute_ratios().
    meta_out = dict(result["meta"])
    meta_out["check"] = result["check"]
    with open("data/meta.json", "w") as f:
        json.dump(meta_out, f, indent=2, default=str)

    print("\n-> data/series.csv")
    print("-> data/harmonized_grid.csv")
    print("-> data/meta.json")