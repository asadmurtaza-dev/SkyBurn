"""
SkyBurn - Demo data generator
==============================
Generates synthetic, FIRMS-shaped data so that:

  1. the pipeline can be tested without waiting for an archive request
  2. there is a working demo if the real archive does not arrive in time
  3. the harmonization can be verified — because here we KNOW the right answer

This is not real data. Never present it as real. It is a stand-in until the
real FIRMS archive arrives.

The pattern it generates (Punjab-like):
  - an October-November peak every year (crop-residue burning season)
  - MODIS from 2001, VIIRS from 2012
  - VIIRS detects ~2.6x more hotspots (375 m vs 1 km pixels)
  - the season drifts slightly earlier each year
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

# Punjab-ish bbox
WEST, SOUTH, EAST, NORTH = 69.3, 27.7, 75.4, 34.0

END_YEAR = 2026
END_MONTH = 11

# The platforms to simulate, with the real FIRMS operating windows and the
# detection ratio each one has relative to the true amount of burning.
#
# Suomi NPP is the reference at 1.0. MODIS detects far fewer fires because of
# its 1 km pixel; NOAA-20 is a near-twin of S-NPP and so sits close to 1.0.
#
# NOAA-20 overlapping S-NPP from 2018 is the important case: if the pipeline
# simply added the VIIRS platforms together, 2018 would show a fake doubling.
PLATFORMS = {
    "MODIS":        {"start": (2001, 1), "end": (2026, 11), "ratio": 1 / 2.6},
    "VIIRS_SNPP":   {"start": (2012, 1), "end": (2026, 11), "ratio": 1.00,
                     "last_day": 2},     # S-NPP stopped on 2 Nov 2026
    "VIIRS_NOAA20": {"start": (2018, 4), "end": (2026, 11), "ratio": 1.15},
}

REFERENCE = "VIIRS_SNPP"
MODIS_START = PLATFORMS["MODIS"]["start"][0]

# What the harmonization should recover for MODIS -> S-NPP
TRUE_RATIO = round(PLATFORMS[REFERENCE]["ratio"] / PLATFORMS["MODIS"]["ratio"], 2)

# FIRMS product codes, so the demo filenames look like real downloads
FIRMS_CODE = {
    "MODIS": "M-C61",
    "VIIRS_SNPP": "SV-C2",
    "VIIRS_NOAA20": "J1V-C2",
    "VIIRS_NOAA21": "J2V-C2",
}

# Seasonal shape: Oct-Nov peak (index 0 = January)
# Punjab burns twice a year, once after each harvest: wheat residue in
# April-May, rice residue in October-November, with a quiet summer between.
# Index 0 = January.
SEASON = np.array([0.12, 0.10, 0.20, 0.70, 0.80, 0.25,
                   0.15, 0.18, 0.35, 1.00, 0.85, 0.22])

BASE_FIRES = 220        # MODIS-scale hotspots in the peak month, year 1
TREND_PER_YEAR = 0.018  # a slight real increase each year
SEASON_SHIFT_PER_DECADE = 0.4   # months the season drifts earlier per decade


def _season_weight(month: int, year: int) -> float:
    """Seasonal weight, including the drift."""
    decades = (year - MODIS_START) / 10.0
    shift = SEASON_SHIFT_PER_DECADE * decades
    # shifted month index (wrap around 12)
    idx = (month - 1 + shift) % 12
    lo, hi = int(np.floor(idx)) % 12, int(np.ceil(idx)) % 12
    frac = idx - np.floor(idx)
    return float(SEASON[lo] * (1 - frac) + SEASON[hi] * frac)


def _true_activity(year: int, month: int) -> float:
    """The true amount of burning that both sensors are trying to observe."""
    trend = (1 + TREND_PER_YEAR) ** (year - MODIS_START)
    seasonal = _season_weight(month, year)
    noise = RNG.lognormal(0, 0.28)
    return BASE_FIRES * trend * seasonal * noise


def _scatter_points(n: int, year: int, month: int, platform: str,
                    max_day: int = 31) -> pd.DataFrame:
    """Scatter n hotspots across the area."""
    if n <= 0:
        return pd.DataFrame()

    # Fires occur in clusters, not uniformly scattered.
    n_clusters = max(1, int(n / 25))
    cx = RNG.uniform(WEST + 0.3, EAST - 0.3, n_clusters)
    cy = RNG.uniform(SOUTH + 0.3, NORTH - 0.3, n_clusters)
    pick = RNG.integers(0, n_clusters, n)

    lat = np.clip(cy[pick] + RNG.normal(0, 0.18, n), SOUTH, NORTH)
    lon = np.clip(cx[pick] + RNG.normal(0, 0.18, n), WEST, EAST)

    days_in_month = pd.Period(f"{year}-{month:02d}").days_in_month
    day = RNG.integers(1, min(max_day, days_in_month) + 1, n)
    acq_date = [f"{year}-{month:02d}-{d:02d}" for d in day]

    df = pd.DataFrame({
        "latitude": lat.round(4),
        "longitude": lon.round(4),
        "acq_date": acq_date,
        "acq_time": RNG.integers(0, 2359, n),
        "frp": RNG.lognormal(2.2, 0.9, n).round(1),
        "daynight": RNG.choice(["D", "N"], n, p=[0.65, 0.35]),
    })

    if platform == "MODIS":
        df["brightness"] = RNG.normal(330, 20, n).round(1)
        df["confidence"] = RNG.integers(0, 101, n)
        df["instrument"] = "MODIS"
        df["satellite"] = RNG.choice(["Terra", "Aqua"], n)
    else:
        df["bright_ti4"] = RNG.normal(320, 22, n).round(1)
        df["confidence"] = RNG.choice(["l", "n", "h"], n, p=[0.18, 0.62, 0.20])
        df["instrument"] = "VIIRS"
        df["satellite"] = {
            "VIIRS_SNPP": "N", "VIIRS_NOAA20": "N20", "VIIRS_NOAA21": "N21"
        }.get(platform, "N")

    return df


def _in_window(year: int, month: int, window: dict) -> bool:
    ym = year * 12 + month
    lo = window["start"][0] * 12 + window["start"][1]
    hi = window["end"][0] * 12 + window["end"][1]
    return lo <= ym <= hi


def generate() -> dict:
    """One FIRMS-format DataFrame per platform."""
    rows = {p: [] for p in PLATFORMS}
    truth_rows = []

    for year in range(MODIS_START, END_YEAR + 1):
        last_month = END_MONTH if year == END_YEAR else 12
        for month in range(1, last_month + 1):
            truth = _true_activity(year, month)

            # Record the true activity on the reference scale, so the
            # harmonization can afterwards be scored against a known answer.
            truth_rows.append({
                "year": year, "month": month,
                "truth": truth * PLATFORMS[REFERENCE]["ratio"],
            })

            # Every platform observes the SAME underlying fires, each with its
            # own detection rate. This is what makes double counting visible:
            # from 2018 three platforms are watching one set of fires.
            for platform, cfg in PLATFORMS.items():
                if not _in_window(year, month, cfg):
                    continue
                n = RNG.poisson(truth * cfg["ratio"])

                # A platform's final month can be cut short -- S-NPP stopped
                # on 2 November 2026. The pipeline has to notice that rather
                # than reading it as a collapse in burning.
                max_day = 31
                if (year, month) == tuple(cfg["end"]):
                    max_day = cfg.get("last_day", 31)
                    n = int(round(n * max_day / 30)) if max_day < 31 else n

                rows[platform].append(
                    _scatter_points(n, year, month, platform, max_day))

    out = {}
    for platform, chunks in rows.items():
        chunks = [c for c in chunks if not c.empty]
        if chunks:
            out[platform] = pd.concat(chunks, ignore_index=True)
    return out, pd.DataFrame(truth_rows)


if __name__ == "__main__":
    out_dir = Path("data/archive")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating demo data ...")
    frames, truth = generate()
    truth.to_csv("data/truth.csv", index=False)

    for platform, df in frames.items():
        code = FIRMS_CODE[platform]
        path = out_dir / f"fire_archive_{code}_demo.csv"
        df.to_csv(path, index=False)
        span = PLATFORMS[platform]
        print(f"  {platform:14s} {len(df):>9,} hotspots  "
              f"({span['start'][0]}-{span['end'][0]})  -> {path.name}")

    print(f"\n  Reference platform: {REFERENCE}")
    print(f"  True MODIS -> {REFERENCE} ratio injected: {TRUE_RATIO}x")
    print("  (the harmonization should recover a ratio close to this)")
    print("\n  NOAA-20 overlaps S-NPP from 2018. If the pipeline simply added")
    print("  the VIIRS platforms together, 2018 would show a fake doubling —")
    print("  harmonize.py picks one platform per month instead.")
    print("  (the harmonization should recover a ratio close to this)")