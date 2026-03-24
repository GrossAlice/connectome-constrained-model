#!/usr/bin/env python3
"""Aggregate all LOO R² results from the last week."""
import json, re, time as _t
from pathlib import Path
import numpy as np

base = Path("output_plots/stage2")
results = []

# 1) Sweep results JSON files
for jf in base.rglob("sweep_results.json"):
    sweep_name = str(jf.parent.relative_to(base))
    with open(jf) as f:
        data = json.load(f)
    for r in data:
        if "loo_med" in r:
            results.append({
                "source": sweep_name,
                "name": r.get("name", "?"),
                "os_med": r.get("os_med"),
                "loo_med": r.get("loo_med"),
                "loo_mean": r.get("loo_mean"),
                "loo_pos": r.get("loo_pos"),
                "fr_med": r.get("fr_med"),
                "epochs": None,
            })

# 2) Individual run.log files (from last 7 days)
cutoff = _t.time() - 7 * 86400
for rl in base.rglob("run.log"):
    if rl.stat().st_mtime < cutoff:
        continue
    rel = str(rl.parent.relative_to(base))
    if any(x in rel for x in ["sweep_loo/", "sweep_loo2/", "sweep_sensitivity/"]):
        if rel.count("/") > 0 and not rel.endswith(("sweep_loo", "sweep_loo2", "sweep_sensitivity")):
            continue
    try:
        text = rl.read_text()
    except Exception:
        continue
    loo_vals = re.findall(r"loo_progress.*?r2=([0-9eE.+-]+)\s+total=", text)
    if not loo_vals:
        continue
    loo_r2 = [float(x) for x in loo_vals if abs(float(x)) < 100]
    if not loo_r2:
        continue
    ep_match = re.findall(r"ep\s+(\d+)/(\d+)", text)
    ep_str = "{}/{}".format(ep_match[-1][0], ep_match[-1][1]) if ep_match else "?"
    arr = np.array(loo_r2)
    fin = arr[np.isfinite(arr)]
    results.append({
        "source": rel,
        "name": rel,
        "os_med": None,
        "loo_med": round(float(np.median(fin)), 4),
        "loo_mean": round(float(np.mean(fin)), 4),
        "loo_pos": round(float(np.mean(fin > 0)), 4),
        "fr_med": None,
        "epochs": ep_str,
    })

# Sort by LOO median descending
results.sort(key=lambda r: r.get("loo_med") or -999, reverse=True)

# Print
hdr = "{:<60s} {:>7s} {:>8s} {:>9s} {:>7s} {:>7s} {:>8s}".format(
    "Source/Run", "OS med", "LOO med", "LOO mean", "LOO>0%", "FR med", "epochs"
)
print(hdr)
print("─" * 110)
for r in results[:50]:
    src = r["source"]
    nm = r["name"]
    label = "{}/{}".format(src, nm) if src != nm else nm
    if len(label) > 58:
        label = "..." + label[-55:]
    print("{:<60s} {:>7s} {:>8s} {:>9s} {:>7s} {:>7s} {:>8s}".format(
        label,
        str(r.get("os_med", "") or ""),
        str(r.get("loo_med", "") or ""),
        str(r.get("loo_mean", "") or ""),
        str(r.get("loo_pos", "") or ""),
        str(r.get("fr_med", "") or ""),
        str(r.get("epochs", "") or ""),
    ))

print("\nTotal runs with LOO data: {}".format(len(results)))
