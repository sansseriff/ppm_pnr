"""
Photon-number figures, from pnr_refit.py (photon-number model with fitted cluster weights and timing offset):

  groupings figure (manuscript Fig. 6; the model's components are in Fig. 5a, make_fig_structured_gmm.py)
    a) joint slot + photon-number attribution of PPM detections, referenced to the true slot
       (grey: copies of the coloured data translated by one slot; dashed: |2> / |3> overlap of adjacent slots)
    b) the same, only for detections attributed to the correct slot
  statistics figure (manuscript Fig. 7)
    a) / b) measured photon-number fractions M_j (vacuum and attributed photon numbers) vs Poisson P_j(mu),
            log and linear scale

Needs <dataset>/inter/pnr_refit.json (pnr_refit.py) and <dataset>/figs/ppm_decoding_<tag>_table.json
(make_fig_decoding.py, for mu).

usage: uv run python -m analysis.make_fig_pnr --dataset 20   (or 10)
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import orjson
from matplotlib.patches import Ellipse

from analysis import style
from analysis.datasets import decode_path, get
from analysis.display import offset_for_center
from analysis.pnr_refit import joint_attribution, load_groups, load_stream_events, poisson_pmf

GROUP_COLORS = ["#1f5fa8", "#7b2fd6", "#e8338f", "#f59a45", "#9be05a"]
labels_for = lambda J: [str(j) for j in range(1, J)] + [f"{J}+"]  # photon numbers of J groups: 1 .. J-1, J+


def groupings_figure(cfg):
    T = cfg["slot_ps"]
    groups = load_groups(cfg["model"])
    J = len(groups)
    r = json.loads(open(cfg["refit"]).read())[f"{float(cfg['group_db']):.1f}"]
    v = np.asarray(r["shift"])
    ev = load_stream_events(decode_path(cfg, cfg["group_db"]), T)
    slot, num = joint_attribution(ev, groups, np.asarray(r["weights"]), v, T)
    rel = ev["xy"] - T * ev["true"][:, None]
    ok = slot == ev["true"]
    print(f"{cfg['tag']} {cfg['group_db']} dB: slot accuracy {ok.mean():.4f} on {len(ok)} detections; "
          f"photon-number fractions {np.round(np.bincount(num, minlength=J) / len(num), 3).tolist()}")

    # display coordinates (analysis/display.py)
    off = offset_for_center(cfg, np.average(groups[0]["means"], weights=groups[0]["weights"], axis=0) + v)
    rel = rel + off
    v = v + off

    # the model's components are shown in Fig. 5a (make_fig_structured_gmm.py)
    fig, axs = plt.subplots(1, 2, figsize=(9, 9))
    kw = dict(s=3, rasterized=True)
    for i in range(-3, 4):
        if i:
            axs[0].scatter(rel[:, 0] + i * T, rel[:, 1] + i * T, color="black", alpha=0.02, **kw)
    for j in range(J):
        sel = num == j
        axs[0].scatter(rel[sel, 0], rel[sel, 1], color=GROUP_COLORS[j], alpha=0.08, label=rf"$M_{{{labels_for(J)[j]}}}$", **kw)
        sel &= ok
        axs[1].scatter(rel[sel, 0], rel[sel, 1], color=GROUP_COLORS[j], alpha=0.08, label=rf"$M_{{C{labels_for(J)[j]}}}$", **kw)

    if cfg["key"] == "20":
        # |2> events of slot i overlap |3> events of slot i+1 (as marked in the 2023 figure)
        axs[0].add_patch(Ellipse((-43, 405), 20, 60, angle=-45, fill=False, ls="--", color="black", lw=1.5))
    if cfg["key"] == "20":
        xlim, ylim = (-150, 50), (235, 600)  # as in the 2023 figure
    else:
        com = np.array([np.average(g["means"], weights=g["weights"], axis=0) for g in groups]) + v
        lo, hi = com.min(axis=0), com.max(axis=0)
        xlim, ylim = (lo[0] - 1.3 * T, hi[0] + 1.3 * T), (lo[1] - 1.3 * T, hi[1] + 1.3 * T)
    for ax, lab in zip(axs, "ab"):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$t_A$ (ps)")
        ax.text(-0.1, 1.06, f"{lab})", transform=ax.transAxes, fontsize=19, va="top")
        lg = ax.legend(numpoints=1, fontsize=14, frameon=False, loc="lower right")
        for h in lg.legend_handles:
            h.set_sizes([52]), h.set_alpha(0.6)
    axs[0].set_ylabel(r"$t_B$ (ps)")
    style.save(fig, cfg["out"] + f"pnr_resolved_arrival_time_groupings_{cfg['tag']}")
    return dict(slot_accuracy=float(ok.mean()), n=int(len(ok)), mu=r["mu"], shift=r["shift"])


def statistics_figure(cfg):
    refit = json.load(open(cfg["refit"]))
    dbs = sorted(int(float(k)) for k in refit)
    mu = np.array([refit[f"{d:.1f}"]["mu"] for d in dbs])
    M = np.array([refit[f"{d:.1f}"]["measured_with_vacuum"] for d in dbs])  # (n_db, J+1)
    N = np.array([refit[f"{d:.1f}"]["n_missing"] + refit[f"{d:.1f}"]["n_valid"] for d in dbs])
    J = M.shape[1] - 1
    fine = np.geomspace(mu.min() * 0.8, mu.max() * 1.2, 400)
    P = np.array([poisson_pmf(m, J) for m in fine])
    colors = ["#222222"] + GROUP_COLORS
    markers = ["o", "D", "^", "s", "<", ">"]
    names = ["0"] + labels_for(J)

    fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(19, 6))
    for ax in (ax_log, ax_lin):
        for j in range(J + 1):
            err = 2 * np.sqrt(M[:, j] * N) / N  # 2-sigma counting error, as in the 2023 figure
            ax.plot(fine, P[:, j], color=colors[j], alpha=0.7, label=rf"$P_{{{names[j]}}}$")
            ax.plot(mu, M[:, j], markers[j], color=colors[j], label=rf"$M_{{{names[j]}}}$")
            ax.errorbar(mu, M[:, j], yerr=err, fmt="o", color=colors[j], elinewidth=3.5, capsize=0, alpha=0.5)
        ax.set_ylim(3e-5, 1.1)
        ax.set_xlim(1.5e-2 if cfg["key"] == "20" else 0.8 * mu.min(), 10)
        ax.set_ylabel("Probability")
        ax.set_xlabel(r"Mean photon number $\mu$")
    ax_log.set_xscale("log"), ax_log.set_yscale("log")
    ax_lin.legend(loc=(0.6, 0.18) if cfg["key"] == "20" else (0.62, 0.24), ncol=2, fancybox=False, fontsize=14)
    for ax, lab in zip((ax_log, ax_lin), "ab"):
        ax.text(-0.1, 1.06, f"{lab})", transform=ax.transAxes, fontsize=19, va="top")
    style.save(fig, cfg["out"] + f"pnr_stats_{cfg['tag']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["10", "20"], required=True)
    ap.add_argument("--model", default=None, help="default: <dataset>/inter/static_sim.json")
    ap.add_argument("--refit", default=None, help="default: <dataset>/inter/pnr_refit.json")
    ap.add_argument("--out-dir", default=None, help="default: <dataset>/figs")
    args = ap.parse_args()
    cfg = get(args.dataset)
    cfg["model"] = args.model or cfg["model"]
    cfg["refit"] = args.refit or cfg["refit"]
    out_dir = args.out_dir or str(cfg["figs"])
    cfg["out"] = out_dir + os.sep
    os.makedirs(out_dir, exist_ok=True)
    style.apply()
    info = groupings_figure(cfg)
    statistics_figure(cfg)
    with open(cfg["out"] + f"pnr_figures_{cfg['tag']}.json", "w") as f:
        json.dump(info, f, indent=1)


if __name__ == "__main__":
    main()
