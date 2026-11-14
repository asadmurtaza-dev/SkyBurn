import io
import json
import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import harmonize
import analytics

app = Flask(__name__, static_folder="static", static_url_path="")


app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
  
    resp.headers.pop("ETag", None)
    resp.headers.pop("Last-Modified", None)
    return resp

GRID_PATH = Path("data/harmonized_grid.csv")
META_PATH = Path("data/meta.json")

# held in memory
GRID = None
META = {}


def load_grid():
    """Load the harmonized grid, or explain how to build it."""
    global GRID, META

    if not GRID_PATH.exists():
        raise SystemExit(
            f"{GRID_PATH} not found.\n"
            "Run these first:\n"
            "  python3 make_demo_data.py   (or add a real FIRMS archive)\n"
            "  python3 harmonize.py"
        )

    GRID = pd.read_csv(GRID_PATH)
    print(f"  grid loaded: {len(GRID):,} cell-months, "
          f"{GRID['cell'].nunique():,} cells, "
          f"{int(GRID['year'].min())}-{int(GRID['year'].max())}")

    META = {
        "cells": int(GRID["cell"].nunique()),
        "year_min": int(GRID["year"].min()),
        "year_max": int(GRID["year"].max()),
    }

    if META_PATH.exists():
        with open(META_PATH) as f:
            saved = json.load(f)
        for key in ("global_ratio", "global_ratios", "reference", "platforms",
                    "overlaps", "spans", "cells_calibrated"):
            if key in saved:
                META[key] = saved[key]

        spans = saved.get("spans") or {}
        ref = saved.get("reference")
        if ref and ref in spans:
            META["viirs_start"] = spans[ref][0]

        ratios = saved.get("global_ratios") or {}
        detail = ", ".join(f"{p} x{r}" for p, r in ratios.items() if p != ref)
        print(f"  reference: {ref or 'unknown'}")
        if detail:
            print(f"  calibration: {detail}")
    else:
        print(f"  [!] {META_PATH} not found -- re-run harmonize.py")
        META["global_ratio"] = None

    return GRID


def parse_bbox():
    """?bbox=west,south,east,north  -> tuple or None"""
    raw = request.args.get("bbox", "").strip()
    if not raw:
        return None
    try:
        parts = [float(x) for x in raw.split(",")]
    except ValueError:
        return None
    if len(parts) != 4:
        return None

    w, s, e, n = parts
    # swap if given back to front
    if w > e:
        w, e = e, w
    if s > n:
        s, n = n, s
    return (w, s, e, n)



#  Routes


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/report")
def api_report():
    """
    The full report for one area.
    ?bbox=69.3,27.7,75.4,34.0
    ?metric=harmonized | raw      (raw is only for the side-by-side comparison)
    """
    bbox = parse_bbox()
    metric = request.args.get("metric", "harmonized")
    if metric not in ("harmonized", "raw"):
        metric = "harmonized"

    series = harmonize.area_series(GRID, bbox)

    if series.empty:
        return jsonify({
            "error": "No hotspots found in this area. "
                     "Try a larger area or a different location.",
            "bbox": bbox,
        }), 404


    check = harmonize.discontinuity_check(series, meta=META)
    report = analytics.full_report(series, META, check, value_col=metric)
    report["bbox"] = bbox
    report["metric"] = metric

    # send both curves so the frontend can toggle between them
    report["series"] = {
        "labels": [f"{int(r.year)}-{int(r.month):02d}" for r in series.itertuples()],
        "harmonized": [float(v) for v in series["harmonized"]],
        "raw": [float(v) for v in series["raw"]],
        "modis": [float(v) for v in series["modis"]],
        "viirs": [float(v) for v in series["viirs"]],
    }

    return jsonify(report)


@app.route("/api/cells")
def api_cells():
    """
    Where the burning actually was, for one month.

    Returns the harmonized count per 0.25-degree grid cell so the map can draw
    the spatial pattern. Cells rather than individual hotspots: a busy month
    holds tens of thousands of detections, which is slow to draw and no easier
    to read, and the harmonized value is defined per cell-month anyway.

    ?bbox=69.3,27.7,75.4,34.0&year=2026&month=11
    """
    bbox = parse_bbox()

    try:
        year = int(request.args["year"])
        month = int(request.args["month"])
    except (KeyError, ValueError):
        return jsonify({"error": "year and month are required"}), 400

    if not 1 <= month <= 12:
        return jsonify({"error": "month must be 1-12"}), 400

    d = GRID[(GRID["year"] == year) & (GRID["month"] == month)]

    if bbox:
        w, s, e, n = bbox
        d = d[
            (d["cell_lon"] >= w) & (d["cell_lon"] <= e)
            & (d["cell_lat"] >= s) & (d["cell_lat"] <= n)
        ]

    d = d[d["harmonized"] > 0]

    cells = [
        {
            "lat": round(float(r.cell_lat), 4),
            "lon": round(float(r.cell_lon), 4),
            "v": int(r.harmonized),
        }
        for r in d.itertuples()
    ]
    cells.sort(key=lambda c: c["v"])      # draw the busiest last, on top

    return jsonify({
        "year": year,
        "month": month,
        "cell_size": harmonize.GRID_DEG,
        "cells": cells,
        "total": int(d["harmonized"].sum()),
        "max": int(d["harmonized"].max()) if not d.empty else 0,
    })


@app.route("/api/export")
def api_export():
    """CSV export for the selected area and period."""
    bbox = parse_bbox()
    series = harmonize.area_series(GRID, bbox)

    if series.empty:
        return jsonify({"error": "no data to export"}), 404

    out = series[["year", "month", "harmonized", "raw", "modis", "viirs"]].copy()
    out = out.rename(columns={
        "harmonized": "hotspots_harmonized",
        "raw": "hotspots_raw_unadjusted",
        "modis": "hotspots_modis_only",
        "viirs": "hotspots_viirs_only",
    })

    buf = io.StringIO()
    buf.write("# SkyBurn - harmonized burning activity\n")
    buf.write("# Data: NASA FIRMS (MODIS + VIIRS active fire hotspots)\n")
    buf.write("# hotspots_harmonized = VIIRS-equivalent counts\n")
    buf.write(f"# area bbox (w,s,e,n): {bbox if bbox else 'full dataset'}\n")
    buf.write("#\n")
    out.to_csv(buf, index=False)

    data = io.BytesIO(buf.getvalue().encode("utf-8"))
    return send_file(
        data,
        mimetype="text/csv",
        as_attachment=True,
        download_name="skyburn_burning_activity.csv",
    )


@app.route("/api/health")
def api_health():
    return jsonify({
        "ok": GRID is not None,
        "meta": META,
    })


print("SkyBurn starting ...")
load_grid()


if __name__ == "__main__":
    host = os.environ.get("SKYBURN_HOST", "0.0.0.0")

    port = int(os.environ.get("PORT")
               or os.environ.get("SKYBURN_PORT")
               or 5000)

    print(f"\n  -> http://localhost:{port}\n")
    app.run(host=host, port=port, debug=False)
