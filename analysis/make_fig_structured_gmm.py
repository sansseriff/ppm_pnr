"""
Fig. 5: the photon-number-structured Gaussian mixture model (20 GHz).

  a) the model built from the 24 dB preamble (analysis/ica_gmm.py): 3-sigma ellipses of its components, coloured by
     photon-number group, on the preamble detections, with the photon-number boundaries of Fig. 4
  b) the PPM decision rule at 26 dB: the model as used by the decoder (components refitted to the 26 dB preamble in
     the same ICA bases, group weights and timing offset calibrated on that preamble; fair_comparison.py --model
     local), translated by multiples of the 50 ps slot. Contours: density of each slot; shading: the region where
     that slot is the most probable
  c) the multi-photon components along the round-2 photon-number axis (mean +- 1 sigma), on the histogram of the
     preamble detections; dashed: the valleys that assign components to photon numbers

Times are in display coordinates (analysis/display.py).

usage: uv run python -m analysis.make_fig_structured_gmm
"""

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter1d

from analysis import style
from analysis.datasets import decode_path, get
from analysis.display import display_offset
from analysis.fair_comparison import load_decode, structured_gm
from analysis.ica_gmm import Basis, geometry_path, preamble_points
from analysis.make_fig_gmm_intro import XLIM, YLIM, slot_regions
from analysis.make_fig_ica_space import LINE_COLOR, boundary_segment
from analysis.make_fig_pnr import GROUP_COLORS
from analysis.pnr_refit import load_groups

RULE_DB = 26  # attenuation of the decision-rule panel (mu = 2.27)


def labels_for(J):
    return [str(j) for j in range(1, J)] + [f"{J}+"]


def ellipse(ax, m, c, color, lw=1.4, alpha=0.9):
    u, s, _ = np.linalg.svd(c)
    ax.add_patch(Ellipse(m, 6 * np.sqrt(s[0]), 6 * np.sqrt(s[1]), angle=np.degrees(np.arctan2(u[1, 0], u[0, 0])),
                         fill=False, edgecolor=color, lw=lw, alpha=alpha))


def main():
    cfg = get("20")
    T = cfg["slot_ps"]
    mu = {r["db"]: r["mu"] for r in json.loads(open(cfg["table"]).read())["rows"]}
    groups = load_groups(cfg["model"])
    J = len(groups)
    geo = json.load(open(geometry_path(cfg["model"])))
    b1 = Basis(c=geo["basis1"]["c"], W=geo["basis1"]["W"])
    b2 = Basis(c=geo["basis2"]["c"], W=geo["basis2"]["W"])
    edges = geo["boundaries"]
    x = preamble_points(cfg)
    off = display_offset(cfg, cfg["model_db"])

    style.apply()
    fig = plt.figure(figsize=(12, 10.5))
    ax_a = fig.add_axes([0.07, 0.36, 0.40, 0.60])
    ax_b = fig.add_axes([0.56, 0.36, 0.40, 0.60])
    ax_c = fig.add_axes([0.07, 0.06, 0.89, 0.20])

    # a) components of the model on the preamble
    xd = x + off
    h, xe, ye = np.histogram2d(xd[:, 0], xd[:, 1], bins=(np.arange(-250, 150), np.arange(150, 900)))
    ax_a.imshow(np.ma.masked_less_equal(h.T, 0), origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
                cmap="viridis", norm=mpl.colors.LogNorm(), alpha=0.5, rasterized=True)
    for basis, z0 in [(b1, geo["cut"])] + [(b2, e) for e in edges]:
        seg = boundary_segment(basis, z0) + off
        ax_a.plot(seg[:, 0], seg[:, 1], color=LINE_COLOR, ls="--", lw=1.0, alpha=0.4)
    for j, g in enumerate(groups):
        for m, c in zip(g["means"] + off, g["covs"]):
            ellipse(ax_a, m, c, GROUP_COLORS[j])
        ax_a.plot([], [], color=GROUP_COLORS[j], lw=2, label=rf"$C_{{{labels_for(J)[j]}}}$")
    ax_a.legend(frameon=False, loc="lower right", fontsize=13)
    ax_a.text(XLIM[0] + 15, YLIM[1] - 20, style.bold(f"components: {sum(len(g['weights']) for g in groups)}"),
              fontsize=15, usetex=style.USETEX)
    ax_a.text(XLIM[0] + 15, YLIM[1] - 45, style.bold(rf"\mu = {mu[cfg['model_db']]:.2f}"), fontsize=15,
              usetex=style.USETEX)

    # b) decision rule at RULE_DB: the model as the decoder uses it
    gm, _ = structured_gm(load_decode(decode_path(cfg, RULE_DB)), cfg["model"], local=True)
    gmd = dict(means=np.asarray(gm.means) + display_offset(cfg, RULE_DB), covs=np.asarray(gm.covariances),
               weights=np.asarray(gm.weights))
    label_colors = slot_regions(ax_b, gmd, T)
    for k, name in zip(range(-3, 2), ["i-3", "i-2", "i-1", "i", "i+1"]):
        ax_b.text(-104 + (k + 2) * T, 485 + (k + 2) * T, style.bold(name), ha="center", va="center", fontsize=14,
                  color=label_colors[k + 4], usetex=style.USETEX)
    ax_b.text(XLIM[0] + 15, YLIM[1] - 20, style.bold(rf"\mu = {mu[RULE_DB]:.2f}"), fontsize=15, usetex=style.USETEX)

    for ax in (ax_a, ax_b):
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$t_A$ (ps)")
    ax_a.set_ylabel(r"$t_B$ (ps)")

    # c) components along the photon-number axis
    one = b1.to_z(x)[:, 0] > geo["cut"]
    z2 = b2.to_z(x[~one])[:, 0]
    hh, e = np.histogram(z2, bins=np.arange(np.percentile(z2, 0.05), np.percentile(z2, 99.95), 0.5))
    smooth = gaussian_filter1d(hh.astype(float), 1.5)
    ax_c.plot(e[:-1], smooth, color="black", lw=1.8, alpha=0.6)
    top = smooth.max()
    x_hi = e[-1]
    for j, g in enumerate(groups[1:], start=1):
        mz = b2.to_z(g["means"])[:, 0]
        sz = np.sqrt(np.einsum("i,kij,j->k", b2.W[:, 0], g["covs"], b2.W[:, 0]))
        order = np.argsort(mz)
        for r, (m_, s_) in enumerate(zip(mz[order], sz[order])):
            ax_c.errorbar(m_, top * (0.08 + 0.12 * r), xerr=s_, fmt="o", color=GROUP_COLORS[j], ms=5, capsize=0, lw=2)
            x_hi = max(x_hi, m_ + 1.5 * s_)
    for e_ in edges:
        ax_c.axvline(e_, color=LINE_COLOR, ls="--", lw=1.2, alpha=0.7)
    ax_c.set_xlim(e[0], x_hi)
    ax_c.set_ylim(0, top * 1.1)
    ax_c.set_xlabel(r"photon-number feature, round 2 ($|1\rangle$ masked)")
    ax_c.set_ylabel("counts")

    for ax, lab, xy in [(ax_a, "a)", (-0.18, 1.04)), (ax_b, "b)", (-0.18, 1.04)), (ax_c, "c)", (-0.07, 1.15))]:
        ax.text(*xy, lab, transform=ax.transAxes, fontsize=19, va="top")
    style.save(fig, cfg["figs"] / "structured_gmm")


if __name__ == "__main__":
    main()
