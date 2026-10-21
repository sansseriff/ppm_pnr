"""
Fig. S1 (supplement): linear (ICA) projection of the two-threshold response that separates photon numbers
(20 GHz preamble data, 24 dB).

The projection is the ICA unmixing matrix from a FastICA fit made in Nov 2023
(ppm_20ghz/ica_unmixing_nov2023.json), normalized to unit determinant and rotated by -3 degrees, a
rotation chosen by eye to sharpen the projected peaks. A FastICA refit of the same data with current
scikit-learn (any seed) converges to a different direction (~144 deg) that resolves only two peaks, so
the stored matrix is used to reproduce the published figure. Any projection direction between ~60 and
~120 degrees resolves the five photon-number peaks; the stored one is at ~103.5 degrees.

usage: uv run python -m analysis.make_fig_ica
"""

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import orjson

from analysis.datasets import decode_path, get
from analysis.display import display_offset

DB = 24
ROTATION_DEG = -3.0
UNMIXING = "ica_unmixing_nov2023.json"


def rotation(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def main():
    cfg = get("20")
    mu = {r["db"]: r["mu"] for r in json.loads(open(cfg["table"]).read())["rows"]}
    pts = np.asarray(orjson.loads(open(decode_path(cfg, DB), "rb").read())["hist_data"]["counts"], float)
    centered = pts - pts.mean(axis=0)
    keep = (np.abs(centered) < 250).all(axis=1)
    xc = centered[keep]
    disp = pts[keep] + display_offset(cfg, DB)

    vec = np.array(json.loads((cfg["root"] / UNMIXING).read_text()))
    w = (vec / np.sqrt(abs(np.linalg.det(vec)))) @ rotation(ROTATION_DEG)
    proj = xc @ w

    mpl.rcParams.update({"font.size": 10, "axes.linewidth": 0.6, "mathtext.fontset": "dejavusans"})
    fig = plt.figure(figsize=(11, 4))
    ax1 = fig.add_axes([0.06, 0.13, 0.25, 0.80])
    ax2 = fig.add_axes([0.40, 0.56, 0.57, 0.36])
    ax3 = fig.add_axes([0.40, 0.13, 0.57, 0.36], sharex=ax2)

    ax1.hist2d(disp[:, 0], disp[:, 1], bins=(np.arange(-400, 300), np.arange(0, 900)), norm=mpl.colors.LogNorm(),
               density=True, cmap="viridis", rasterized=True)
    # directions of the two unmixing vectors (dotted), through the centre of the data
    c = disp.mean(axis=0)
    rot90 = np.array([[0, -1], [1, 0]])
    for col, color in [(0, "red"), (1, "purple")]:
        d = rot90 @ w[:, col]
        d = d / np.linalg.norm(d)
        ax1.plot(c[0] + np.array([-250, 250]) * d[0], c[1] + np.array([-250, 250]) * d[1], color=color, lw=2,
                 ls=":", alpha=0.35)
    ax1.set_xlim(c[0] - 100, c[0] + 140)  # same window around the data centre as the published figure
    ax1.set_ylim(c[1] - 150, c[1] + 240)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel(r"$t_A$ (ps)")
    ax1.set_ylabel(r"$t_B$ (ps)")
    ax1.text(0.06, 0.94, rf"$\mathbf{{\mu = {mu[DB]:.2f}}}$", transform=ax1.transAxes, va="center", fontsize=13)
    for x, y, n in [(40, 190, "1"), (-20, 50, "2"), (-48, -5, "3"), (-62, -36, "4"), (-78, -69, "5+")]:
        ax1.text(c[0] + x, c[1] + y, rf"$\mathbf{{|{n}\rangle}}$", ha="center", va="center", fontsize=12)

    ax2.hist2d(proj[:, 0], proj[:, 1], bins=[np.arange(-300, 300, 1), np.arange(-300, 300, 2)],
               norm=mpl.colors.LogNorm(), density=True, cmap="viridis", rasterized=True)
    ax2.set_xlim(-75, 125)
    ax2.set_ylim(-100, 100)
    ax2.set_ylabel("ICA feature space y")
    hist, edges = np.histogram(proj[:, 0], bins=np.arange(-70, 130, 0.5))
    ax3.plot(edges[:-1], hist, color="black", lw=1.8, alpha=0.4)
    ax3.set_xlabel("ICA feature space x")
    ax3.set_ylabel("counts")
    for x, y, n in [(88.5, 500, "1"), (22, 1010, "2"), (-7, 1100, "3"), (-22, 1100, "4"), (-39, 700, "5+")]:
        ax3.text(x, y, rf"$\mathbf{{|{n}\rangle}}$", ha="center", va="center", fontsize=11)
    for ax, lab, xy in [(ax1, "a)", (-0.25, 1.08)), (ax2, "b)", (-0.08, 1.19)), (ax3, "c)", (-0.08, 1.05))]:
        ax.text(*xy, lab, transform=ax.transAxes, fontsize=16, va="top")

    out = cfg["figs"] / "ICA_component_analysis_light"
    fig.savefig(f"{out}.pdf", dpi=300)
    fig.savefig(f"{out}.png", dpi=110)
    print(f"-> {out}.pdf")


if __name__ == "__main__":
    main()
