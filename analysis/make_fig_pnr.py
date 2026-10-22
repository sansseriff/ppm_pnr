"""
Photon-number figures, from pnr_refit.py (Poisson-weighted photon-number model with fitted timing offset):

  groupings figure (manuscript Fig. 6)
    a) gaussian components of the photon-number model, coloured by group C_j
    b) joint slot + photon-number attribution of PPM detections, referenced to the true slot
       (grey: copies of the coloured data translated by one slot; dashed: |2> / |3> overlap of adjacent slots)
    c) the same, only for detections attributed to the correct slot
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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import orjson
from matplotlib.patches import Ellipse

from analysis.datasets import decode_path, get
from analysis.display import offset_for_center
from analysis.pnr_refit import joint_attribution, load_groups, load_stream_events, poisson_pmf

GROUP_COLORS = ["#1f5fa8", "#7b2fd6", "#e8338f", "#f59a45", "#9be05a"]
LABELS = ["1", "2", "3", "4", "5+"]


def draw_groups(ax, groups, v):
    for j, g in enumerate(groups):
        wmax = g["weights"].max()
        for m, c, w in zip(g["means"] + v, g["covs"], g["weights"]):
            vals, vecs = np.linalg.eigh(c)
            ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
            e = Ellipse(m, 2 * np.sqrt(vals[1]), 2 * np.sqrt(vals[0]), angle=ang, fill=False,
                        color=GROUP_COLORS[j], lw=1.4, alpha=0.25 + 0.75 * w / wmax)
            ax.add_patch(e)
        ax.plot([], [], color=GROUP_COLORS[j], lw=2, label=rf"$C_{{{LABELS[j]}}}$")


def groupings_figure(cfg):
    T = cfg["slot_ps"]
    groups = load_groups(cfg["model"])
    J = len(groups)
    r = json.loads(open(cfg["refit"]).read())[f"{float(cfg['group_db']):.1f}"]
    v = np.asarray(r["shift"])
    prior = poisson_pmf(r["mu"], J)[1:]
    ev = load_stream_events(decode_path(cfg, cfg["group_db"]), T)
    slot, num = joint_attribution(ev, groups, prior, v, T)
    rel = ev["xy"] - T * ev["true"][:, None]
    ok = slot == ev["true"]
    print(f"{cfg['tag']} {cfg['group_db']} dB: slot accuracy {ok.mean():.4f} on {len(ok)} detections; "
          f"photon-number fractions {np.round(np.bincount(num, minlength=J) / len(num), 3).tolist()}")

    # display coordinates (analysis/display.py)
    off = offset_for_center(cfg, np.average(groups[0]["means"], weights=groups[0]["weights"], axis=0) + v)
    rel = rel + off
    v = v + off

    fig, axs = plt.subplots(1, 3, figsize=(12, 6.2), sharex=True, sharey=True)
    draw_groups(axs[0], groups, v)
    kw = dict(s=2.0, lw=0, rasterized=True)
    for i in range(-6, 7):
        if i:
            axs[1].scatter(rel[:, 0] + i * T, rel[:, 1] + i * T, color="#555555", alpha=0.04, **kw)
    for j in range(J):
        sel = num == j
        axs[1].scatter(rel[sel, 0], rel[sel, 1], color=GROUP_COLORS[j], alpha=0.25, label=rf"$M_{{{LABELS[j]}}}$", **kw)
        sel &= ok
        axs[2].scatter(rel[sel, 0], rel[sel, 1], color=GROUP_COLORS[j], alpha=0.25, label=rf"$M_{{C{LABELS[j]}}}$", **kw)

    if cfg["key"] == "20":
        # |2> events of slot i overlap |3> events of slot i+1 (as marked in the 2023 figure)
        c2 = np.average(groups[1]["means"], weights=groups[1]["weights"], axis=0) + v
        axs[1].add_patch(Ellipse(c2 + np.array([0, -3]), 46, 22, angle=40, fill=False, ls="--", color="black", lw=1.2))
    com = np.array([np.average(g["means"], weights=g["weights"], axis=0) for g in groups]) + v
    lo, hi = com.min(axis=0), com.max(axis=0)
    pad = 1.3 * T
    for ax, lab in zip(axs, "abc"):
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$t_A$ (ps)")
        ax.grid(alpha=0.2)
        ax.text(-0.12, 1.0, f"{lab})", transform=ax.transAxes, fontsize=12, va="top")
        lg = ax.legend(frameon=False, loc="lower right", fontsize=8, markerscale=5)
        for h in lg.legend_handles:
            h.set_alpha(0.9)
    axs[0].set_ylabel(r"$t_B$ (ps)")
    fig.tight_layout()
    out = cfg["out"] + f"pnr_resolved_arrival_time_groupings_{cfg['tag']}_light"
    fig.savefig(out + ".pdf", dpi=300)
    fig.savefig(out + ".png", dpi=110)
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
    names = ["0"] + LABELS

    fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax in (ax_log, ax_lin):
        for j in range(J + 1):
            err = np.sqrt(M[:, j] * (1 - M[:, j]) / N)
            ax.plot(fine, P[:, j], color=colors[j], lw=1.6, label=rf"$P_{{{names[j]}}}$")
            ax.errorbar(mu, M[:, j], yerr=err, fmt=markers[j], color=colors[j], ms=4.5, capsize=0,
                        elinewidth=3, ecolor=mpl.colors.to_rgba(colors[j], 0.35), label=rf"$M_{{{names[j]}}}$")
        ax.set_xlabel(r"mean photon number $\mu$")
        ax.grid(alpha=0.2)
    ax_log.set_xscale("log"), ax_log.set_yscale("log")
    ax_log.set_ylim(3e-5, 1.3)
    ax_log.set_xlim(fine[0], fine[-1])
    ax_log.set_ylabel("probability")
    ax_lin.set_xlim(0, 10)
    ax_lin.set_ylim(0, 1.02)
    h, l = ax_lin.get_legend_handles_labels()
    order = [i for pair in zip(range(0, 2 * (J + 1), 2), range(1, 2 * (J + 1), 2)) for i in pair]
    ax_lin.legend([h[i] for i in order], [l[i] for i in order], ncol=2, fontsize=8, frameon=True, loc="center right")
    for ax, lab in zip((ax_log, ax_lin), "ab"):
        ax.text(-0.1, 1.0, f"{lab})", transform=ax.transAxes, fontsize=12, va="top")
    fig.tight_layout()
    out = cfg["out"] + f"pnr_stats_{cfg['tag']}_light"
    fig.savefig(out + ".pdf", dpi=300)
    fig.savefig(out + ".png", dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["10", "20"], required=True)
    args = ap.parse_args()
    cfg = get(args.dataset)
    cfg["out"] = str(cfg["figs"]) + os.sep
    os.makedirs(cfg["figs"], exist_ok=True)
    mpl.rcParams.update({"font.size": 9, "axes.linewidth": 0.6, "mathtext.fontset": "dejavusans"})
    info = groupings_figure(cfg)
    statistics_figure(cfg)
    with open(cfg["out"] + f"pnr_figures_{cfg['tag']}.json", "w") as f:
        json.dump(info, f, indent=1)


if __name__ == "__main__":
    main()
