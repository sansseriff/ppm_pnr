"""
Experimental (branch ica-gmm): N / seed sweep of the ICA-space models of analysis/ica_gmm.py, evaluated as the
published model is (group weights + timing shift refit on the group_db dataset, joint slot + photon-number
attribution), with photon-number fractions against Poisson.

usage: DS=10 GROUPING=axis uv run python -m analysis.ica_gmm_sweep   (GROUPING: louvain | axis)
outputs: <dataset>/inter/ica_gmm/  models, sweep_<grouping>.json, sweep_<grouping>.png
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt
from analysis import style
from analysis.datasets import decode_path, get
from analysis.display import offset_for_center
from analysis.ica_gmm import build
from analysis.make_fig_pnr import GROUP_COLORS
from analysis.pnr_refit import (flatten, fit_weights_and_shift, init_shift_grid, joint_attribution, load_events,
                                load_groups, load_stream_events, poisson_pmf)

KEY = os.environ.get("DS", "10"); GROUPING = os.environ.get("GROUPING", "louvain")
cfg = get(KEY); T = cfg["slot_ps"]; db = cfg["group_db"]
S = cfg["inter"] / "ica_gmm"
S.mkdir(exist_ok=True)
mu = {r["db"]: r["mu"] for r in json.loads(open(cfg["table"]).read())["rows"]}[db]
x, n_missing, n_total = load_events(decode_path(cfg, db), T)
ev = load_stream_events(decode_path(cfg, db), T)
span = np.ptp(x, axis=0); bg = -np.log(span[0] * span[1])
pois_for = lambda J: poisson_pmf(mu, J)[1:] / poisson_pmf(mu, J)[1:].sum()


def evaluate(path):
    groups = load_groups(path)
    means, _, inner_w, _ = flatten(groups)
    center = np.full(2, (np.median(x, axis=0) - np.average(means, weights=inner_w, axis=0)).mean())
    v0 = init_shift_grid(x, groups, center, bg, half_width=600.0, step=5.0)
    pi, v, ll = fit_weights_and_shift(x, groups, v0, bg)
    J = len(groups)
    w = pi[:J] / pi[:J].sum()
    slot, num = joint_attribution(ev, groups, w, v, T)
    return groups, v, ll, slot == ev["true"], num


configs = [(r, n, s) for r in ((1, 2) if GROUPING == "louvain" else (2,)) for n in (8, 12, 16, 20, 25, 30) for s in (42, 7)]
results = []
for rounds, n, seed in configs:
    out = f"{S}/{GROUPING}_r{rounds}_n{n}_s{seed}.json"
    try:
        groups, info = build(cfg, rounds, n, 3, seed, GROUPING)
    except RuntimeError as e:
        print(rounds, n, seed, "FAILED", e, flush=True); continue
    json.dump(groups, open(out, "w"))
    groups, v, ll, ok, num = evaluate(out)
    frac = np.bincount(num, minlength=len(groups)) / len(num)
    k = sum(len(g["weights"]) for g in groups)
    results.append(dict(rounds=rounds, n=n, seed=seed, k=k, ll=ll, acc=float(ok.mean()), frac=frac.tolist(),
                        freq=info["partition_freq"], path=out))
    print(f"rounds={rounds} N={n:2d} seed={seed:2d} comps={k:2d}  loglik {ll:9.0f}  slot acc {ok.mean():.4f}  "
          f"fractions {np.round(100 * frac, 1).tolist()}  Poisson {np.round(100 * pois_for(len(groups)), 1).tolist()}  freq {info['partition_freq']:.2f}", flush=True)
json.dump(dict(results=results), open(f"{S}/sweep_{GROUPING}.json", "w"), indent=1)

# figure: correct-slot attribution for seed 42, one row per round count
sel = [r for r in results if r["seed"] == 42]
style.apply()
ns = sorted({r["n"] for r in sel})
rows = sorted({r["rounds"] for r in sel})
fig, axs = plt.subplots(len(rows), len(ns), figsize=(3.0 * len(ns), 4.5 * len(rows)), squeeze=False)
lims = ((-230, 110), (290, 760)) if KEY == "10" else ((-160, 60), (235, 610))
for r in sel:
    a = axs[rows.index(r["rounds"]), ns.index(r["n"])]
    groups, v, ll, ok, num = evaluate(r["path"])
    off = offset_for_center(cfg, np.average(groups[0]["means"], weights=groups[0]["weights"], axis=0) + v)
    rel = ev["xy"] - T * ev["true"][:, None] + off
    for j in range(len(groups)):
        m = (num == j) & ok
        a.scatter(rel[m, 0], rel[m, 1], color=GROUP_COLORS[j], alpha=0.08, s=3, rasterized=True)
    a.set_title(f"{r['rounds']} ICA round(s), N={r['n']} ({r['k']} comps)\n"
                + " ".join(f"{100 * f:.0f}" for f in r["frac"]) + " %", fontsize=10)
    a.set_xlim(*lims[0]); a.set_ylim(*lims[1]); a.set_aspect("equal")
fig.savefig(f"{S}/sweep_{GROUPING}.png", dpi=80, bbox_inches="tight")
print(f"{S}/sweep_{GROUPING}.png")
