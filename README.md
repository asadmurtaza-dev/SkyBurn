# SkyBurn

**NASA Space Apps Challenge 2026 — Harmonization of MODIS and VIIRS Hot Spots**

A harmonized burning activity calendar. The MODIS and VIIRS active fire records
are rescaled onto one consistent series, so a change in the graph means a change
in the fires, not a change in the satellite.

---

## Quick start

```bash
# 1. Virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Dependencies
pip install -r requirements.txt

# 3. Settings
cp .env.example .env               # the FIRMS key goes in later

# 4. Demo data (lets you test without waiting for the real archive)
python3 make_demo_data.py

# 5. Harmonize
python3 harmonize.py

# 6. Run
python3 app.py
#    -> http://localhost:5000
```

> On Windows use `python`, not `python3`.

### What goes in .env

| Variable | Purpose | Required? |
|---|---|---|
| `FIRMS_MAP_KEY` | Fetching recent data from the FIRMS API | Only for live data. Not needed for the archive. |
| `SKYBURN_PORT` | Server port (default 5000) | No |
| `SKYBURN_HOST` | Server host (default 0.0.0.0) | No |

`.env` is git-ignored — an API key must not end up in the repository. Only
`.env.example`, an empty template, is committed.

## Using real FIRMS data

The demo data exists only to exercise the pipeline. For the real thing:

1. **Archive** (the historical record) —
   https://firms.modaps.eosdis.nasa.gov/download/

   Sign in with an Earthdata account or an emailed code. Choose **CSV** as the
   format. Request these sources:

   | Source | FIRMS code | Date range |
   |---|---|---|
   | MODIS C6.1 | `M-C61` | `2000-11-01` to today |
   | VIIRS S-NPP 375m | `SV-C2` | `2012-01-20` to 2026-11-02 |
   | VIIRS NOAA-20 375m | `J1V-C2` | `2018-04-01` to today |

   S-NPP stopped producing new data on 2 November 2026, so NOAA-20 is what
   covers the present day.

   A download link arrives by email. Each zip holds a `fire_archive_*` file
   (the historical record) and a `fire_nrt_*` file (the most recent weeks).
   **Both belong in `data/archive/`** — together they cover the record right up
   to today, so the API is not needed.

   The loader reads FIRMS product codes directly, so the downloaded filenames
   work unchanged.

   **As soon as the files arrive, run:**
   ```bash
   python3 check_archive.py
   ```
   It validates every file and explains what to fix. The critical check is the
   **overlap** between MODIS and VIIRS — without it there is no way to derive
   the calibration ratio.

2. **API** (recent data — the archive excludes the last 7 days):
   ```bash
   # free key: firms.modaps.eosdis.nasa.gov/api/map_key/
   # put it in .env as FIRMS_MAP_KEY=your_key
   python3 firms_fetch.py --area punjab_pk --days 5
   ```

3. Then re-run `python3 harmonize.py`.

---

## How the harmonization works

**The problem.** A MODIS pixel is 1 km across; a VIIRS pixel is 375 m. VIIRS
detects smaller fires, so it reports far more hotspots — **for the same fires**.

And it is not only MODIS vs VIIRS. Several VIIRS satellites flew at once —
Suomi NPP (2012–2026), NOAA-20 (2018–) and NOAA-21 (2023–). Adding them
together counts the same fire two or three times, and creates a fresh
artificial step every time a satellite launches.

**The approach.** Treat the record as a set of *platforms* and put every
platform on the scale of one reference platform (Suomi NPP — it overlaps MODIS
at one end and the NOAA satellites at the other).

1. Measure each platform's ratio to the reference over the period they both flew
2. Give each 0.25° cell its own ratio, shrunk toward the global one
3. For each month use exactly **one** platform — never a sum
4. Rescale that platform's count onto the reference

### Three subtle mistakes this code avoids

**1. Selection bias in the ratio.** Filtering cell-months on "at least N
detections in both" keeps only the months where the weaker sensor happened to
score well, biasing the ratio downward. Fix: total the whole overlap period
instead.

**2. The same bias at cell level.** Keeping only cells with `counts >= 20`
filters on the ratio's *denominator*, which does the same thing. Measured
against a known 2.6 ratio, that filter returned 2.41. Fix: no threshold —
every cell is kept and shrunk toward the global ratio by how little data it has.

**3. Jensen's inequality.** A per-cell ratio divides two noisy counts, and the
expected value of such a ratio sits above the true one. Multiply independent
counts by those ratios and the rebuilt era comes out high — the MODIS era
landed **17% above truth**. Fix: per-cell ratios decide *where* the activity
sits, the global ratio decides *how much*, by rescaling each era's total.

And two that are not subtle:

**Decide by era, not by row.** Inside the reference's era, a month where the
reference saw nothing must read zero, even if another satellite saw something.
The reference is the measurement.

**A stopped satellite is not a quiet month.** Suomi NPP began on 20 January
2012 and stopped on 2 November 2026, so each of those months holds only a few
days of data. Read as whole months they look like burning collapsed — and the
last month is what the "current conditions" panel reports, so the error lands
in the most prominent number on the page. Those boundary months are dropped,
and the month falls through to the next satellite. A platform still flying is
left alone: its short final month is just the present month, which the
analytics pro-rate.

**When substituting, prefer the least rescaling.** If the reference is
unavailable for a month, the substitute is the platform needing the smallest
scale factor — a sister VIIRS satellite at 0.87x, not MODIS at 2.67x — because
every unit of rescaling is a unit of assumption. Ranking by record length
instead would pick MODIS for the present month purely because it has flown
longer, which is exactly backwards.

### Evidence that it works

`make_demo_data.py` simulates three satellites watching the same fires, with
known detection rates. Because the answer is known, `validate.py` can grade the
result rather than just inspect it:

```
BIAS (harmonized / truth, 1.00 is perfect)      harmonized      raw
  overall                                            1.003    1.338
  2001-2011  MODIS only                              1.009    0.386
  2012-2017  S-NPP                                   0.998    1.363
  2018-2026  S-NPP + NOAA-20                         0.999    2.497   <- double counting

STEPS AT TRANSITIONS                  truth   harmonized      raw
  2012  MODIS -> S-NPP                +18.8%      +15.2%  +326.9%
  2018  NOAA-20 joins S-NPP            +6.1%       +3.9%   +95.4%

AGREEMENT WITH TRUTH
  harmonized   r = 0.983   mean abs error = 12.9%
  raw          r = 0.759   mean abs error = 86.0%
```

The raw record is 2.5x too high once two VIIRS satellites fly. The harmonized
record sits within 1% of the truth in every era. Run it yourself:

```bash
python3 make_demo_data.py && python3 harmonize.py && python3 validate.py
```

## What the app shows

| Section | What it tells you |
|---|---|
| **Current conditions** | The latest month against the median of the same month in previous years. A partial month is pro-rated and labelled as a projection. |
| **Burning activity calendar** | Year × month heatmap, scaled to the 95th percentile so one extreme month does not flatten the rest. Sequential blue ramp, validated for colour-vision deficiency. |
| **Raw vs harmonized** | Both curves together — the evidence that the harmonization did something. |
| **Fire season** | Which months form the season, when it peaks, what share of the year it accounts for, and whether the pattern looks agricultural or wildfire-driven. |
| **Season shift** | How far the season has drifted across the record, using circular statistics (January follows December). |
| **Map** | Where the burning was, as a heat grid over the selected area. Clicking any month in the calendar paints that month on the map. |
| **CSV export** | The selected area and period, with the source and method recorded in the header. |

---

## Files

```
check_archive.py    run this first when the real FIRMS files arrive
validate.py         grades the harmonization against the demo's known truth
firms_fetch.py      FIRMS API + archive loader, column normalisation
harmonize.py        calibration ratios, harmonized counts, discontinuity test
analytics.py        calendar, status, season, season shift
app.py              Flask API
static/index.html   frontend (single file)
static/vendor/      Leaflet (local copy, so it works without a CDN)
make_demo_data.py   synthetic test data with a known ground truth
requirements.txt    dependencies
.env.example        settings template (copy to .env)
.gitignore          keeps .env and data/ out of git
```

## Demo notes

- The app opens on Punjab. The calendar shows a dark October–November band in
  every year — crop-residue burning, the season that drives Lahore's smog. The
  hackathon falls inside that season.
- Spend time on the **raw vs harmonized** chart. That is Part 1 of the challenge,
  and it is where most submissions will stop at simply concatenating the files.
- Say plainly that the ratio method is an approximation and not an official NASA
  product. Reviewers respond well to stated limitations.
- Map tiles come from the internet. Without a connection the map disappears but
  **the app keeps working** — presets and coordinate entry still select an area.

## Credits

Hotspot data: **NASA FIRMS** (Fire Information for Resource Management System) —
MODIS (Terra/Aqua) and VIIRS (Suomi NPP, NOAA-20, NOAA-21) active fire products.

Harmonized counts are VIIRS-equivalent estimates produced by rescaling the
MODIS-only era using sensor-overlap ratios. They are an approximation, not an
official NASA product.
