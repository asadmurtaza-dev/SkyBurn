"""
SkyBurn - Archive checker
==========================
Run this FIRST, as soon as the real FIRMS CSVs arrive:

    python3 check_archive.py

It reports whether the files are usable and, when something is wrong, what to
do about it. Much better to find out now than at 2am during the hackathon.
"""

import sys
from pathlib import Path

import pandas as pd

from firms_fetch import platform_from_name, platform_from_columns, VIIRS_PLATFORMS

ARCHIVE = Path("data/archive")

# Columns a FIRMS CSV must have
NEEDED = ["latitude", "longitude", "acq_date"]
NICE = ["acq_time", "confidence", "frp", "daynight", "instrument"]

OK = "  [ok] "
WARN = "  [!]  "
BAD = "  [X]  "


def detect_platform(path: Path, df: pd.DataFrame) -> str:
    """
    Identify the satellite. Prefer the filename, fall back to the contents.

    A trailing '?' means the filename gave nothing and this was inferred,
    which cannot reliably tell the VIIRS satellites apart.
    """
    platform = platform_from_name(path.stem)
    if platform:
        return platform
    guess = platform_from_columns(df)
    return (guess + "?") if guess else "UNKNOWN"


def check_file(path: Path) -> dict:
    print(f"\n{path.name}")
    print("  " + "-" * (len(path.name)))

    try:
        df = pd.read_csv(path, nrows=50000)
    except Exception as e:
        print(BAD + f"Could not read the file: {e}")
        return {"ok": False}

    full_rows = sum(1 for _ in open(path, encoding="utf-8", errors="ignore")) - 1
    print(OK + f"{full_rows:,} rows")

    # --- columns ---
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        print(BAD + f"Required columns are missing: {missing}")
        print("       Columns present: " + ", ".join(df.columns[:12]))
        return {"ok": False}
    print(OK + "required columns present")

    absent = [c for c in NICE if c not in df.columns]
    if absent:
        print(WARN + f"Optional columns missing: {absent} (this still works)")

    # --- platform ---
    platform = detect_platform(path, df)
    if platform == "UNKNOWN":
        print(BAD + "Could not determine which satellite this is.")
        print("       Rename the file to include the FIRMS product code or the")
        print("       satellite name, e.g. M-C61, SV-C2, J1V-C2, modis, viirs.")
        return {"ok": False, "platform": None}
    if platform.endswith("?"):
        clean = platform.rstrip("?")
        print(WARN + f"Filename gave no clue; the contents look like {clean}.")
        print("       If this is NOAA-20 or NOAA-21 rather than Suomi NPP,")
        print("       rename the file to say so — otherwise the two get merged")
        print("       and the same fires are counted twice.")
        platform = clean
    else:
        print(OK + f"platform: {platform}")

    # --- dates ---
    dates = pd.to_datetime(df["acq_date"], errors="coerce")
    bad_dates = int(dates.isna().sum())
    if bad_dates:
        print(WARN + f"{bad_dates} rows have an unparseable date (they will be skipped)")

    # date range across the whole file
    try:
        d_all = pd.read_csv(path, usecols=["acq_date"])
        d_all = pd.to_datetime(d_all["acq_date"], errors="coerce").dropna()
        lo, hi = d_all.min(), d_all.max()
        years = hi.year - lo.year + 1
        print(OK + f"dates: {lo.date()} to {hi.date()} ({years} years)")
    except Exception:
        lo = hi = None
        years = 0

    # --- coordinates ---
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    if lat.between(-90, 90).mean() < 0.99 or lon.between(-180, 180).mean() < 0.99:
        print(BAD + "latitude/longitude values are out of range")
        return {"ok": False}
    print(OK + f"area: lat {lat.min():.1f}..{lat.max():.1f}, "
               f"lon {lon.min():.1f}..{lon.max():.1f}")

    # --- confidence format ---
    if "confidence" in df.columns:
        c = df["confidence"]
        is_num = pd.to_numeric(c, errors="coerce").notna().mean()
        if is_num > 0.9:
            print(OK + "confidence: numeric (MODIS style)")
        else:
            vals = sorted(set(c.astype(str).str.lower().str.strip()))[:6]
            print(OK + f"confidence: letters (VIIRS style) {vals}")

    return {"ok": True, "platform": platform, "rows": full_rows,
            "lo": lo, "hi": hi, "path": path}


def main():
    print("=" * 62)
    print("SkyBurn - archive check")
    print("=" * 62)

    if not ARCHIVE.exists():
        print(BAD + f"The folder {ARCHIVE}/ does not exist.")
        print("\n  Create it and put the FIRMS CSVs inside:")
        print(f"      mkdir -p {ARCHIVE}")
        return 1

    # De-duplicate: Windows filesystems are case-insensitive, so *.csv and
    # *.CSV both match the same file and every file would be listed twice.
    seen = {}
    for pattern in ("*.csv", "*.CSV"):
        for f in ARCHIVE.glob(pattern):
            seen[f.resolve()] = f
    files = sorted(seen.values())

    if not files:
        print(BAD + f"{ARCHIVE}/ is empty -- no CSV files found.")
        print("\n  Request an archive download from FIRMS:")
        print("      https://firms.modaps.eosdis.nasa.gov/download/")
        print("  Then put the CSVs here and run this script again.")
        return 1

    print(f"\nFound {len(files)} file(s) in {ARCHIVE}/")

    results = [check_file(f) for f in files]
    good = [r for r in results if r.get("ok")]

    # ---- verdict ----
    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)

    if not good:
        print(BAD + "No usable files. Fix the problems listed above.")
        return 1

    platforms = {}
    for r in good:
        platforms.setdefault(r["platform"], []).append(r)

    spans = {}
    for p, rs in sorted(platforms.items()):
        total = sum(r["rows"] for r in rs)
        lo = min(r["lo"] for r in rs if r["lo"] is not None)
        hi = max(r["hi"] for r in rs if r["hi"] is not None)
        spans[p] = (lo, hi)
        print(f"  {p:14s} {total:>10,} hotspots   {lo.date()} .. {hi.date()}")

    problems = []

    has_modis = "MODIS" in platforms
    viirs_present = [p for p in platforms if p in VIIRS_PLATFORMS]

    if not has_modis:
        problems.append(
            "No MODIS data. The long historical record comes from MODIS.\n"
            "       Request 'MODIS C6.1' from FIRMS (2000-11-01 to today)."
        )
    if not viirs_present:
        problems.append(
            "No VIIRS data. Calibration needs a modern sensor to tie to.\n"
            "       Request 'VIIRS S-NPP 375m' from FIRMS (2012-01-20 onward)."
        )

    # Which platform everything will be expressed in
    reference = None
    for candidate in ("VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21", "MODIS"):
        if candidate in platforms:
            reference = candidate
            break

    if reference:
        print(f"\n  Reference platform: {reference}")
        print("  Every other platform is rescaled onto this one.")

    # overlap check -- the most important one
    if reference:
        ref_lo, ref_hi = spans[reference]
        print()
        for p, (lo, hi) in sorted(spans.items()):
            if p == reference:
                continue
            ov_lo, ov_hi = max(lo, ref_lo), min(hi, ref_hi)
            ov_days = (ov_hi - ov_lo).days

            if ov_days <= 0:
                problems.append(
                    f"{p} never overlaps {reference}.\n"
                    "       Without an overlap there is no way to calibrate it.\n"
                    "       Re-request with wider date ranges."
                )
            elif ov_days < 365:
                print(WARN + f"{p}: only {ov_days} days overlapping {reference} "
                             f"({ov_lo.date()} .. {ov_hi.date()})")
                print("       Works, but the ratio will be less reliable.")
            else:
                print(OK + f"{p}: {ov_days / 365.25:.1f} years overlapping "
                           f"{reference} -- enough to calibrate")

        # the MODIS-only era is the part that gains the most
        if has_modis and reference != "MODIS":
            m_lo = spans["MODIS"][0]
            if m_lo < ref_lo:
                solo = (ref_lo - m_lo).days / 365.25
                print(OK + f"MODIS-only era: {solo:.1f} years "
                           f"({m_lo.date()} .. {ref_lo.date()})")
                print(f"       This is the part rescaled to {reference}-equivalent.")
            else:
                print(WARN + "There is no MODIS-only era in this data.")
                print("       Harmonization will add little. Re-request MODIS "
                      "starting 2000-11-01.")

    # Warn about overlapping VIIRS satellites — the double-counting case
    if len(viirs_present) > 1:
        print()
        print(OK + f"{len(viirs_present)} VIIRS satellites present: "
                   f"{', '.join(sorted(viirs_present))}")
        print("       They flew at the same time, so adding them together would")
        print("       count the same fires twice. harmonize.py uses one platform")
        print("       per month to avoid that.")

    if problems:
        print()
        for p in problems:
            print(BAD + p)
        print("\n  Fix these and run this script again.")
        return 1

    print("\n" + OK + "All good. Now run:")
    print("\n      python3 harmonize.py")
    print("      python3 app.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())