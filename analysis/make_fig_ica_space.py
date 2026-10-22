"""
Fig. 4: photon-number feature space from independent component analysis (20 GHz preamble, 24 dB).

  a) preamble detections in (t_A, t_B) with the photon-number boundaries mapped back from the ICA feature spaces:
     the |1> cut of round 1 and the valleys of round 2 (lines of constant photon-number feature)
  b) round 1: projection of all preamble detections on the photon-number axis; the |1> cluster is cut off at the
     empty gap
  c) round 2: ICA refitted without the |1> detections; the valleys between the photon-number peaks are the group
     boundaries of the photon-number model (analysis/ica_gmm.py)

Times are in display coordinates (analysis/display.py).

usage: uv run python -m analysis.make_fig_ica_space
"""

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from snsphd.layout import bisect

from analysis import style
from analysis.datasets import get
from analysis.display import display_offset
from analysis.ica_gmm import build, valley_boundaries

LINE_COLOR = "#d62728"


def boundary_segment(basis, z0, t=(-400, 400)):
    """(t_A, t_B) points of the line of constant photon-number feature z0 in this basis"""
    ts = np.array(t, float)
    return basis.c + z0 * basis.M[0] + ts[:, None] * basis.M[1]


def projection_panel(ax, z0, cuts, labels, label_pos):
    h, e = np.histogram(z0, bins=np.arange(np.percentile(z0, 0.05), np.percentile(z0, 99.95), 0.5))
    smooth = gaussian_filter1d(h.astype(float), 1.5)
    ax.plot(e[:-1], smooth, color="black", lw=1.8, alpha=0.6)
    for c in cuts:
        ax.axvline(c, color=LINE_COLOR, ls="--", lw=1.2, alpha=0.7)
    for x, n in zip(label_pos, labels):
        i = np.clip(np.searchsorted(e, x), 0, len(smooth) - 1)
        ax.text(x, smooth[max(0, i - 4):i + 5].max() + 0.04 * smooth.max(), style.ket(n), ha="center", va="bottom",
                fontsize=12, usetex=style.USETEX)
    ax.set_ylim(0, smooth.max() * 1.25)
    ax.set_xlim(e[0], e[-1])
    ax.set_ylabel("counts")


def main():
    cfg = get("20")
    db = cfg["model_db"]
    mu = {r["db"]: r["mu"] for r in json.loads(open(cfg["table"]).read())["rows"]}[db]
    _, info = build(cfg, rounds=2, n=20, seed=42, grouping="axis", sigma_max=5.0)
    g = info["_geometry"]
    b1, b2, x, one = g["basis1"], g["basis2"], g["x"], g["one"]
    z1 = b1.to_z(x)[:, 0]
    z2 = b2.to_z(x[~one])[:, 0]
    edges = valley_boundaries(z2)
    off = display_offset(cfg, db)

    style.apply()
    fig = plt.figure(figsize=(11, 4.6))
    left, right = bisect([0, 0, 1, 1], direction="vert", offset=0.3, spacing=0.10)
    right_bottom, right_top = bisect(right, direction="horiz", offset=0.5, spacing=0.13)
    ax1, ax2, ax3 = fig.add_axes(left), fig.add_axes(right_top), fig.add_axes(right_bottom)

    # a) preamble in (t_A, t_B) with the boundaries
    xd = x + off
    ax1.hist2d(xd[:, 0], xd[:, 1], bins=(np.arange(-400, 300), np.arange(0, 900)), norm=mpl.colors.LogNorm(),
               alpha=0.8, density=True, cmap="viridis", rasterized=True)
    for basis, z0 in [(b1, g["cut"])] + [(b2, e) for e in edges]:
        seg = boundary_segment(basis, z0) + off
        ax1.plot(seg[:, 0], seg[:, 1], color=LINE_COLOR, ls="--", lw=1.2, alpha=0.6)
    # photon-number labels at the centre of each interval
    intervals = np.searchsorted(-np.asarray(edges), -z2)
    centers = [np.median(xd[one], axis=0)] + [np.median(xd[~one][intervals == i], axis=0)
                                              for i in range(len(edges) + 1)]
    names = [str(j) for j in range(1, len(edges) + 2)] + [f"{len(edges) + 2}+"]
    for c, n in zip(centers, names):
        ax1.text(c[0] - 42, c[1] + 4, style.ket(n), ha="center", va="center", fontsize=13, usetex=style.USETEX)
    c1 = centers[0]
    ax1.set_xlim(c1[0] - 185, c1[0] + 60)
    ax1.set_ylim(c1[1] - 320, c1[1] + 60)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel(r"$t_A$ (ps)")
    ax1.set_ylabel(r"$t_B$ (ps)")
    ax1.text(0.06, 0.95, style.bold(rf"\mu = {mu:.2f}"), transform=ax1.transAxes, va="top", fontsize=15,
             usetex=style.USETEX)

    # b) round 1, c) round 2
    # round 1: the five most prominent peaks are the photon numbers 5+ .. 1 (as Fig. S1 of the earlier version)
    h, e = np.histogram(z1, bins=np.arange(np.percentile(z1, 0.05), np.percentile(z1, 99.95), 0.5))
    smooth = gaussian_filter1d(h.astype(float), 1.5)
    pk, prop = find_peaks(smooth, prominence=0.03 * smooth.max())
    pk = np.sort(pk[np.argsort(-prop["prominences"])][:5])
    projection_panel(ax2, z1, [g["cut"]], ["5+", "4", "3", "2", "1"][-len(pk):], e[pk])
    ax2.set_xlabel("photon-number feature, round 1 (all detections)")
    pos = [np.median(z2[intervals == i]) for i in range(len(edges) + 1)]
    projection_panel(ax3, z2, edges, names[1:], pos)
    ax3.set_xlabel(r"photon-number feature, round 2 ($|1\rangle$ masked)")

    for ax, lab, xy in [(ax1, "a)", (-0.1, 1.08)), (ax2, "b)", (-0.05, 1.2)), (ax3, "c)", (-0.05, 1.2))]:
        ax.text(*xy, lab, transform=ax.transAxes, fontsize=19, va="top")
    style.save(fig, cfg["figs"] / "ica_feature_space")
    print(f"round-1 cut {g['cut']:.1f}, round-2 valleys {np.round(edges, 1).tolist()}, "
          f"projection angles {np.round(info['projection_angles_deg'], 1).tolist()} deg")


if __name__ == "__main__":
    main()
