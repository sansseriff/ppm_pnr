"""
Fig. 4: Gaussian-mixture model of the two-threshold detector response (20 GHz preamble data).

  a), b) 2D histograms of preamble (slot 0) detections with the 3-sigma ellipses of the GMM components
         fit by the decoder, at a low and a moderate mean photon number
  c)     GMM response for slots i-4 ... i+4 (the model translated by multiples of the 50 ps slot):
         contours of each slot's density and, shaded, the region where each slot is the most probable

Times are in display coordinates (analysis/display.py); mu is the attenuator-calibrated mean photon number.

usage: uv run python -m analysis.make_fig_gmm_intro
"""

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import orjson
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp

from analysis import style
from analysis.datasets import decode_path, get
from analysis.display import display_offset

# (attenuation dB, index into gm_list) for panels a, b, c; gm_list[k] has 5 + 2k components
PANELS = {"a": (36, 1), "b": (26, 7), "c": (26, 6)}
XLIM, YLIM = (-183, 61), (235, 650)
SLOTS = range(-4, 5)
SLOT_CMAPS = ["Reds", "Blues", "Oranges", "Purples", "Greens", "Reds", "Blues", "Oranges", "Purples"]
REGION_COLORS = ["#FF9191", "#9391FF", "#FFCC91", "#DC91FF", "#91FF9C", "#FF9191", "#9391FF", "#FFCC91", "#DC91FF"]


def gmm_logpdf(xy, means, covs, weights):
    inv = np.linalg.inv(covs)
    d = xy[:, None, :] - means[None]
    maha = np.einsum("nki,kij,nkj->nk", d, inv, d)
    lc = -0.5 * maha - 0.5 * np.linalg.slogdet(covs)[1] - np.log(2 * np.pi) + np.log(weights)
    return logsumexp(lc, axis=1)


def load(cfg, db, gm_idx):
    d = orjson.loads(open(decode_path(cfg, db), "rb").read())
    off = display_offset(cfg, db)
    g = d["gm_data"]["gm_list"][gm_idx]
    gm = dict(means=np.asarray(g["means"]) + off, covs=np.asarray(g["covariances"]), weights=np.asarray(g["weights"]))
    pts = np.asarray(d["hist_data"]["counts"], float) + off
    return pts, gm, np.asarray(d["hist_data"]["bins"], float)


def data_and_ellipses(ax, pts, gm, bins):
    h, xe, ye = np.histogram2d(pts[:, 0], pts[:, 1], bins=(bins, bins))
    ax.imshow(np.ma.masked_less_equal(h.T, 0), origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
              cmap="viridis", norm=mpl.colors.LogNorm(), alpha=0.5, rasterized=True)
    for m, c in zip(gm["means"], gm["covs"]):
        u, s, _ = np.linalg.svd(c)
        ax.add_patch(Ellipse(m, 6 * np.sqrt(s[0]), 6 * np.sqrt(s[1]), angle=np.degrees(np.arctan2(u[1, 0], u[0, 0])),
                             fill=False, edgecolor="black", lw=1.4, alpha=0.8))


def slot_regions(ax, gm, T):
    xs, ys = np.arange(*XLIM, 0.5), np.arange(*YLIM, 0.5)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    dens = []
    for k in SLOTS:
        z = np.exp(gmm_logpdf(grid - k * T, gm["means"], gm["covs"], gm["weights"])).reshape(xx.shape)
        dens.append(z / z.max())
    dens = np.stack(dens)
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    best = np.argmax(dens, axis=0)
    significant = dens.max(axis=0) >= 2e-4
    for j in range(len(SLOTS)):
        # as in the original notebook: contours in the squared (darkened) slot colormap on a log scale, and the
        # density itself as a single-colour fill
        rgb = plt.get_cmap(SLOT_CMAPS[j])(np.linspace(0, 1, 256))[:, :3]
        norm = mpl.colors.LogNorm(vmin=1e-3, vmax=1)
        ax.contour(xx, yy, dens[j] + 1e-5, levels=np.linspace(0.001, 1, 10)[1:], cmap=ListedColormap(rgb ** 2),
                   norm=norm, alpha=0.6, linewidths=1.0)
        ax.imshow(dens[j] + 1e-5, alpha=np.clip(dens[j], 0, 1) ** 0.8, origin="lower", extent=extent,
                  cmap=ListedColormap([rgb[230], rgb[230]]), norm=norm, rasterized=True)
    for j in range(len(SLOTS)):
        mask = gaussian_filter((significant & (best == j)).astype(float), 0.2)
        ax.imshow(np.ones_like(mask), alpha=0.2 * mask, origin="lower", extent=extent,
                  cmap=ListedColormap([REGION_COLORS[j]]), rasterized=True)
    return [plt.get_cmap(c)(0.9) for c in SLOT_CMAPS]


def main():
    cfg = get("20")
    T = cfg["slot_ps"]
    mu = {r["db"]: r["mu"] for r in json.loads(open(cfg["table"]).read())["rows"]}

    style.apply()
    fig, axs = plt.subplots(1, 3, figsize=(14, 7))
    for ax, key in zip(axs[:2], "ab"):
        db, gm_idx = PANELS[key]
        pts, gm, bins = load(cfg, db, gm_idx)
        data_and_ellipses(ax, pts, gm, bins)
        ax.text(XLIM[0] + 15, YLIM[1] - 20, style.bold(f"components: {len(gm['weights'])}"), fontsize=16,
                usetex=style.USETEX)
        ax.text(XLIM[0] + 15, YLIM[1] - 45, style.bold(rf"\mu = {mu[db]:.2f}"), fontsize=16, usetex=style.USETEX)
        for x, y, n in [(-50, 538, "1"), (-109, 410, "2"), (-141, 332, "3+")]:
            ax.text(x, y, style.ket(n), ha="center", va="center", fontsize=14, usetex=style.USETEX)

    db, gm_idx = PANELS["c"]
    _, gm, _ = load(cfg, db, gm_idx)
    label_colors = slot_regions(axs[2], gm, T)
    for k, name in zip(range(-3, 2), ["i-3", "i-2", "i-1", "i", "i+1"]):
        axs[2].text(-104 + (k + 2) * T, 485 + (k + 2) * T, style.bold(name), ha="center", va="center",
                    fontsize=14, usetex=style.USETEX, color=label_colors[k + 4])

    for ax, lab in zip(axs, "abc"):
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$t_A$ (ps)")
        ax.text(-0.18, 1.06, f"{lab})", transform=ax.transAxes, fontsize=19, va="top")
    axs[0].set_ylabel(r"$t_B$ (ps)")
    style.save(fig, cfg["figs"] / "gmm_intro_analysis_t", png_dpi=100)


if __name__ == "__main__":
    main()
