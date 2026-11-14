"""
SkyBurn - FIRMS data fetcher
=============================
Retrieves MODIS and VIIRS active fire hotspots from NASA FIRMS.

Two routes:
  1. ARCHIVE -> the historical record (2000 to last week). Requested manually
                from https://firms.modaps.eosdis.nasa.gov/download/ — a download
                link arrives by email. Drop the CSVs into data/archive/.
  2. API     -> recent data (last few days). Needs a free MAP_KEY:
                https://firms.modaps.eosdis.nasa.gov/api/map_key/

NOTE: the API returns only a short window per request (FIRMS documents 1-10
days; 5 is the safe assumption). Do not pull 25 years through it — take the
history from the archive and use the API only for "what is burning now".

NOTE: Suomi NPP stopped producing new data on 2 November 2026. Use NOAA-20 or
NOAA-21 for current observations; S-NPP remains valid for the historical record.
"""

import os
import io
import time
from datetime import date, timedelta
from pathlib import Path

import requests
import pandas as pd

# Load settings from a .env file when one is present. If python-dotenv is not
# installed the module still works — the variable just has to come from the
# shell environment instead.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

FIRMS_API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

DATA_DIR = Path("data")
ARCHIVE_DIR = DATA_DIR / "archive"
CACHE_DIR = DATA_DIR / "cache"
for d in (DATA_DIR, ARCHIVE_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# FIRMS source identifiers
SOURCES = {
    # Standard processing — the historical record
    "modis_sp": "MODIS_SP",
    "viirs_snpp_sp": "VIIRS_SNPP_SP",
    "viirs_noaa20_sp": "VIIRS_NOAA20_SP",
    # Near-real-time — the last few days
    "modis_nrt": "MODIS_NRT",
    "viirs_noaa20_nrt": "VIIRS_NOAA20_NRT",
    "viirs_noaa21_nrt": "VIIRS_NOAA21_NRT",
}

# Days per request. FIRMS documents 1-10; 5 is the safe assumption.
MAX_DAY_RANGE = 5


def get_map_key() -> str:
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FIRMS_MAP_KEY not found.\n"
            "  1. Get a free key: https://firms.modaps.eosdis.nasa.gov/api/map_key/\n"
            "  2. Create a .env file:   cp .env.example .env\n"
            "  3. Add the key to it:    FIRMS_MAP_KEY=your_key_here\n"
            "\n"
            "  (or set it in the shell: export FIRMS_MAP_KEY=your_key_here)"
        )
    return key


def fetch_area(source: str, bbox: tuple, start: date, days: int = MAX_DAY_RANGE,
               map_key: str = None, retries: int = 3) -> pd.DataFrame:
    """
    Fetch hotspots for one area and date window.
    bbox = (west, south, east, north)  e.g. Punjab: (69.3, 27.7, 75.4, 34.0)
    """
    map_key = map_key or get_map_key()
    src = SOURCES.get(source, source)
    days = min(days, MAX_DAY_RANGE)
    coords = ",".join(str(round(c, 4)) for c in bbox)

    url = f"{FIRMS_API}/{map_key}/{src}/{coords}/{days}/{start.isoformat()}"

    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            text = r.text.strip()

            # FIRMS error responses are not CSV
            if not text or text.lower().startswith(("invalid", "error", "<")):
                print(f"  [!] {src} {start}: {text[:120]}")
                return pd.DataFrame()

            df = pd.read_csv(io.StringIO(text))
            if df.empty:
                return df
            df["source"] = src
            return df

        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}/{retries}] {e} -- waiting {wait}s")
            time.sleep(wait)

    return pd.DataFrame()


def fetch_range(source: str, bbox: tuple, start: date, end: date,
                map_key: str = None, pause: float = 0.6) -> pd.DataFrame:
    """
    Walk a longer date range in 5-day chunks.

    Intended for recent data (days or weeks). For 25 years of history, use the
    archive download instead.
    """
    map_key = map_key or get_map_key()
    chunks = []
    cur = start

    while cur <= end:
        days = min(MAX_DAY_RANGE, (end - cur).days + 1)
        df = fetch_area(source, bbox, cur, days, map_key)
        if not df.empty:
            chunks.append(df)
        cur += timedelta(days=days)
        time.sleep(pause)  # MAP_KEY limit: 5000 transactions / 10 min

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


# ---------------------------------------------------------------- #
#  Platform identification
# ---------------------------------------------------------------- #
#
# A "platform" is one satellite instrument. They must be kept apart, because
# each one detects a different number of fires and two platforms flying at
# once would otherwise double-count the same fire.
#
# FIRMS names its downloads by product code, not by instrument name, e.g.
#   fire_archive_M-C61_846222.csv    -> MODIS Collection 6.1
#   fire_archive_SV-C2_846224.csv    -> Suomi NPP VIIRS
#   fire_archive_J1V-C2_846223.csv   -> NOAA-20 (JPSS-1) VIIRS
#   fire_archive_J2V-C2_846225.csv   -> NOAA-21 (JPSS-2) VIIRS
#
# Longest codes first so "j1v" is tested before a bare "v".
PLATFORM_CODES = [
    ("j1v",         "VIIRS_NOAA20"),
    ("j2v",         "VIIRS_NOAA21"),
    ("noaa20",      "VIIRS_NOAA20"),
    ("noaa-20",     "VIIRS_NOAA20"),
    ("noaa21",      "VIIRS_NOAA21"),
    ("noaa-21",     "VIIRS_NOAA21"),
    ("sv-c2",       "VIIRS_SNPP"),
    ("sv_c2",       "VIIRS_SNPP"),
    ("snpp",        "VIIRS_SNPP"),
    ("suomi",       "VIIRS_SNPP"),
    ("m-c61",       "MODIS"),
    ("m_c61",       "MODIS"),
    ("m-c6",        "MODIS"),
    ("modis",       "MODIS"),
    ("viirs",       "VIIRS_SNPP"),   # generic VIIRS -> assume S-NPP
]

VIIRS_PLATFORMS = {"VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21"}


def platform_from_name(name: str) -> str:
    """Identify the satellite platform from a filename. '' if unknown."""
    n = name.lower()
    for code, platform in PLATFORM_CODES:
        if code in n:
            return platform
    return ""


def platform_from_columns(df: pd.DataFrame) -> str:
    """
    Fall back to the file's contents when the name gives nothing.

    This cannot tell the VIIRS satellites apart — they share a schema — so it
    returns the generic VIIRS_SNPP and the caller should warn about it.
    """
    if "satellite" in df.columns:
        sats = df["satellite"].astype(str).str.upper().unique()
        joined = " ".join(sats)
        if "N21" in joined or "NOAA-21" in joined or "JPSS-2" in joined:
            return "VIIRS_NOAA21"
        if "N20" in joined or "NOAA-20" in joined or "JPSS-1" in joined:
            return "VIIRS_NOAA20"
        if "N" in joined and "bright_ti4" in df.columns:
            return "VIIRS_SNPP"
    if "bright_ti4" in df.columns:
        return "VIIRS_SNPP"
    if "brightness" in df.columns:
        return "MODIS"
    return ""


def load_archive(pattern: str = "*.csv") -> pd.DataFrame:
    """
    Load every FIRMS CSV sitting in data/archive/.

    Both the fire_archive_* and fire_nrt_* files from a FIRMS download belong
    here: the archive file holds the historical record and the NRT file covers
    the most recent weeks that the archive does not yet include.
    """
    files = sorted(ARCHIVE_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"{ARCHIVE_DIR}/ is empty.\n"
            "Request an archive download from "
            "https://firms.modaps.eosdis.nasa.gov/download/"
        )

    frames = []
    for f in files:
        print(f"  loading {f.name} ...", end=" ")
        df = pd.read_csv(f)

        platform = platform_from_name(f.stem)
        if not platform:
            platform = platform_from_columns(df)
            if platform:
                print(f"[name unclear, read as {platform}] ", end="")
            else:
                print("[SKIPPED - platform unknown]")
                continue

        df["platform"] = platform
        df["source"] = platform
        print(f"{len(df):,} rows  ({platform})")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No usable files in {ARCHIVE_DIR}/.\n"
            "Run: python3 check_archive.py"
        )

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- #
#  Normalisation: MODIS and VIIRS ship different columns
# ---------------------------------------------------------------- #

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bring both sensors onto one common schema.

    Differences handled here:
      - brightness: 'brightness' on MODIS, 'bright_ti4' on VIIRS
      - confidence: a 0-100 number on MODIS, a letter (l/n/h) on VIIRS
      - footprint:  1 km pixel on MODIS, 375 m on VIIRS
    """
    if df.empty:
        return df

    out = pd.DataFrame()
    out["lat"] = df["latitude"].astype(float)
    out["lon"] = df["longitude"].astype(float)
    out["date"] = pd.to_datetime(df["acq_date"], errors="coerce")

    # acq_time drops leading zeros (7 -> 00:07, 848 -> 08:48)
    t = df.get("acq_time", pd.Series(0, index=df.index))
    t = pd.to_numeric(t, errors="coerce").fillna(0).astype(int)
    out["hour"] = (t // 100).clip(0, 23)
    out["minute"] = (t % 100).clip(0, 59)

    # brightness column name depends on the sensor
    if "bright_ti4" in df.columns:
        out["brightness"] = pd.to_numeric(df["bright_ti4"], errors="coerce")
    elif "brightness" in df.columns:
        out["brightness"] = pd.to_numeric(df["brightness"], errors="coerce")
    else:
        out["brightness"] = pd.NA

    out["frp"] = pd.to_numeric(df.get("frp"), errors="coerce")
    out["daynight"] = df.get("daynight", pd.NA)

    # ---- platform and sensor family ----
    # platform = the individual satellite (VIIRS_SNPP, VIIRS_NOAA20, ...),
    #            which is what the harmonization calibrates against.
    # sensor   = the instrument family (MODIS / VIIRS), kept for display.
    if "platform" in df.columns:
        out["platform"] = df["platform"].astype(str)
    else:
        src = df.get("source", pd.Series("", index=df.index)).astype(str)
        instrument = df.get("instrument", pd.Series("", index=df.index)).astype(str)
        guess = (src + " " + instrument).map(platform_from_name)
        out["platform"] = guess.replace("", "MODIS")

    out["sensor"] = out["platform"].where(
        ~out["platform"].isin(VIIRS_PLATFORMS), "VIIRS"
    )
    out["pixel_km"] = out["sensor"].map({"MODIS": 1.0, "VIIRS": 0.375})

    # ---- confidence onto one scale ----
    # MODIS reports 0-100; VIIRS reports l (low) / n (nominal) / h (high).
    # The FIRMS FAQ states these are NOT the same scale, so both are mapped
    # into low/medium/high buckets rather than compared directly.
    conf = df.get("confidence", pd.Series(pd.NA, index=df.index))
    out["confidence"] = _bucket_confidence(conf, out["sensor"])

    out = out.dropna(subset=["date", "lat", "lon"])
    return out.reset_index(drop=True)


def _bucket_confidence(conf, sensor) -> pd.Series:
    conf_str = conf.astype(str).str.strip().str.lower()
    conf_num = pd.to_numeric(conf, errors="coerce")

    result = pd.Series("medium", index=conf.index, dtype=object)

    # VIIRS: letters
    viirs_mask = sensor == "VIIRS"
    result[viirs_mask & conf_str.isin(["l", "low"])] = "low"
    result[viirs_mask & conf_str.isin(["n", "nominal"])] = "medium"
    result[viirs_mask & conf_str.isin(["h", "high"])] = "high"

    # MODIS: 0-100, using the breakdown FIRMS itself suggests
    # (<30 low, 30-80 nominal, >80 high)
    modis_mask = sensor == "MODIS"
    result[modis_mask & (conf_num < 30)] = "low"
    result[modis_mask & (conf_num >= 30) & (conf_num <= 80)] = "medium"
    result[modis_mask & (conf_num > 80)] = "high"

    return result


# ---------------------------------------------------------------- #

PRESETS = {
    # west, south, east, north
    "punjab_pk":  (69.3, 27.7, 75.4, 34.0),
    "pakistan":   (60.9, 23.6, 77.1, 37.1),
    "california": (-124.5, 32.5, -114.0, 42.0),
    "amazon":     (-74.0, -18.0, -44.0, 5.0),
}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="SkyBurn FIRMS fetcher")
    p.add_argument("--area", default="punjab_pk", help="a preset name or 'w,s,e,n'")
    p.add_argument("--days", type=int, default=5, help="how many days back")
    p.add_argument("--out", default="data/recent.csv")
    args = p.parse_args()

    bbox = PRESETS.get(args.area)
    if bbox is None:
        bbox = tuple(float(x) for x in args.area.split(","))

    end = date.today()
    start = end - timedelta(days=args.days - 1)

    print(f"Area: {bbox}")
    print(f"Dates: {start} -> {end}\n")

    frames = []
    for src in ["modis_nrt", "viirs_noaa20_nrt", "viirs_noaa21_nrt"]:
        print(f"{src} ...")
        df = fetch_range(src, bbox, start, end)
        print(f"  -> {len(df):,} hotspots")
        if not df.empty:
            frames.append(df)

    if not frames:
        print("\nNothing returned. Check your MAP_KEY.")
        raise SystemExit(1)

    raw = pd.concat(frames, ignore_index=True)
    clean = normalise(raw)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.out, index=False)

    print(f"\nTotal: {len(clean):,} hotspots -> {args.out}")
    print(clean.groupby("sensor").size().to_string())