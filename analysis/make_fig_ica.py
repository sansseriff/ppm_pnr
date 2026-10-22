"""
Fig. S1 (supplement): independent component analysis (ICA) of the two-threshold response, which
separates photon numbers along one independent component (20 GHz preamble data, 24 dB).

ICA is scikit-learn's FastICA with a custom contrast function: a narrow Gaussian kernel,
G(u) = -exp(-a u^2 / 2) / a with a = 32, in place of the default log cosh. FastICA approximates
negentropy by E[G(u)]; the default contrasts weigh the tails of the distribution, so for these multi-peaked
data they converge to a direction dominated by the large gap between the 1-photon cluster and the rest
(only two peaks resolved). The narrow kernel is sensitive to the individual photon-number peaks; the result
is the same for every random initialization and for 24 <= a <= 48 (see notebooks/ica_contrast.ipynb).

usage: uv run python -m analysis.make_fig_ica
"""

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import orjson
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.decomposition import FastICA
from snsphd.layout import bisect

from analysis import style
from analysis.datasets import decode_path, get
from analysis.display import display_offset

DB = 24
CROP_PS = 250  # detections within this distance (per axis) of the mean are used
KERNEL_A = 32.0
LABELS = ["5+", "4", "3", "2", "1"]  # photon numbers of the five peaks, left to right along feature x


def narrow_gaussian_contrast(a):
    """FastICA contrast G(u) = -exp(-a u^2/2)/a: returns G'(u) and the mean of G''(u), as FastICA expects"""

    def g(u):
        e = np.exp(-a * u**2 / 2)
        return u * e, ((1 - a * u**2) * e).mean(axis=-1)

    return g


def negentropy(s, bw=0.05):
    """negentropy of a 1D sample from its histogram (in units of its standard deviation)"""
    s = (s - s.mean()) / s.std()
    h, _ = np.histogram(s, bins=np.arange(-8, 8, bw), density=True)
    p = h[h > 0]
    return 0.5 * np.log(2 * np.pi * np.e) + np.sum(p * np.log(p)) * bw


def fit_ica(xc):
    ica = FastICA(fun=narrow_gaussian_contrast(KERNEL_A), whiten="unit-variance", random_state=0, max_iter=2000,
                  tol=1e-6).fit(xc)
    W = ica.components_.T  # columns: projection vectors of the two independent components
    W = W / np.sqrt(abs(np.linalg.det(W)))  # unit determinant: features in ps-like units, as in the 2023 figure
    k = int(np.argmax([negentropy(xc @ W[:, j]) for j in range(2)]))  # the photon-number component
    W = W[:, [k, 1 - k]]
    if np.percentile(xc @ W[:, 0], 99) < -np.percentile(xc @ W[:, 0], 1):  # 1-photon events on the right
        W[:, 0] = -W[:, 0]
    return W, ica


def main():
    cfg = get("20")
    mu = {r["db"]: r["mu"] for r in json.loads(open(cfg["table"]).read())["rows"]}
    pts = np.asarray(orjson.loads(open(decode_path(cfg, DB), "rb").read())["hist_data"]["counts"], float)
    centered = pts - pts.mean(axis=0)
    keep = (np.abs(centered) < CROP_PS).all(axis=1)
    xc = centered[keep]
    disp = pts[keep] + display_offset(cfg, DB)
    W, _ = fit_ica(xc)
    proj = xc @ W
    mixing = np.linalg.inv(W)  # rows: directions in (t_A, t_B) along which each component varies

    style.apply()
    fig = plt.figure(figsize=(11, 4))
    left, right = bisect([0, 0, 1, 1], direction="vert", offset=0.3, spacing=0.10)
    right_bottom, right_top = bisect(right, direction="horiz", offset=0.5, spacing=0.09)
    ax1 = fig.add_axes(left)
    ax2 = fig.add_axes(right_top)
    ax3 = fig.add_axes(right_bottom)
    ax2.sharex(ax3)

    ax1.hist2d(disp[:, 0], disp[:, 1], bins=(np.arange(-400, 300), np.arange(0, 900)), norm=mpl.colors.LogNorm(),
               alpha=0.8, density=True, cmap="viridis", rasterized=True)
    c = disp.mean(axis=0)
    # lines of constant feature x run along the other component's mixing direction (dotted), and vice versa
    for row, color in [(1, "red"), (0, "purple")]:
        d = mixing[row] / np.linalg.norm(mixing[row])
        ax1.plot(c[0] + np.array([-250, 250]) * d[0], c[1] + np.array([-250, 250]) * d[1], color=color, lw=2,
                 ls=":", alpha=0.27)
    # the two projection vectors (columns of W) as arrows from the centre of the data, as in the 2023 figure:
    # directions only, the one perpendicular to the photon-number line long and pointing right, the other
    # short and pointing up
    d = W.T / np.linalg.norm(W, axis=0)[:, None]
    d[1] *= np.sign(d[1, 0]) or 1
    d[0] *= np.sign(d[0, 1]) or 1
    arrows = d * np.array([[0.7], [2.7]])
    ax1.quiver([c[0], c[0]], [c[1], c[1]], arrows[:, 0], arrows[:, 1], zorder=11, width=0.01, scale=6)
    ax1.set_xlim(c[0] - 100, c[0] + 140)
    ax1.set_ylim(c[1] - 150, c[1] + 240)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel(r"$t_A$ (ps)")
    ax1.set_ylabel(r"$t_B$ (ps)")
    ax1.text(0.06, 0.94, style.bold(rf"\mu = {mu[DB]:.2f}"), transform=ax1.transAxes, va="center", fontsize=16,
              usetex=style.USETEX)
    for x, y, n in [(40, 190, "1"), (-20, 50, "2"), (-48, -5, "3"), (-62, -36, "4"), (-78, -69, "5+")]:
        ax1.text(c[0] + x, c[1] + y, style.ket(n), ha="center", va="center", fontsize=14, usetex=style.USETEX)

    lo, hi = np.percentile(proj[:, 0], [0.2, 99.8])
    lo, hi = lo - 0.1 * (hi - lo), hi + 0.05 * (hi - lo)
    ylo, yhi = np.percentile(proj[:, 1], [0.2, 99.8])
    # bins much wider than the 1 ps time-tag quantization, so the transformed grid does not alias into stripes
    ax2.scatter(proj[:, 0], proj[:, 1], alpha=0.02, rasterized=True, s=4, color="black")
    ax2.hist2d(proj[:, 0], proj[:, 1], bins=[np.arange(lo, hi, 0.5), np.arange(ylo, yhi, 2.5)],
               norm=mpl.colors.LogNorm(), alpha=0.8, density=True, cmap="viridis", rasterized=True)
    ax2.set_ylim(ylo, yhi)
    ax2.set_xlim(lo, hi)
    ax2.set_ylabel("ICA feature space y")
    hist, edges = np.histogram(proj[:, 0], bins=np.arange(lo, hi, 0.5))
    ax3.plot(edges[:-1], hist, color="black", lw=1.8, alpha=0.4)
    ax3.set_xlabel("ICA feature space x")
    ax3.set_ylabel("counts")
    # label the five most prominent peaks (photon numbers 5+ ... 1 from left to right)
    smooth = gaussian_filter1d(hist.astype(float), 2)
    pk, prop = find_peaks(smooth, prominence=smooth.max() * 0.03)
    pk = np.sort(pk[np.argsort(-prop["prominences"])][:5])
    for p, n in zip(pk, LABELS[-len(pk):] if len(pk) < 5 else LABELS):
        ax3.text(edges[p], smooth[p] * 1.08 + 0.03 * smooth.max(), style.ket(n), ha="center", va="bottom",
                 fontsize=12, usetex=style.USETEX)
    ax3.set_ylim(0, smooth.max() * 1.3)
    for ax, lab, xy in [(ax1, "a)", (-0.1, 1.08)), (ax2, "b)", (-0.05, 1.19)), (ax3, "c)", (-0.05, 1.02))]:
        ax.text(*xy, lab, transform=ax.transAxes, fontsize=19, va="top")

    style.save(fig, cfg["figs"] / "ICA_component_analysis")
    ang = [np.degrees(np.arctan2(m[1], m[0])) % 180 for m in mixing]
    print(f"mixing directions: photon-number component {ang[0]:.1f} deg, other {ang[1]:.1f} deg; "
          f"{len(pk)} peaks labelled")


if __name__ == "__main__":
    main()
