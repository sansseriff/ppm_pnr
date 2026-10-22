"""
Experimental (branch ica-gmm): photon-number model built in the ICA feature space.

The ICA projection of the preamble response (Fig. S1) separates photon numbers along one feature axis. Here
the gaussian components are fitted in that space with diagonal covariances, so each component varies
independently along the photon-number axis and the other one and cannot tilt across two photon-number
clusters. The components are then mapped back to (t_A, t_B) (x = z M + c, Sigma_x = M^T Sigma_z M with
M = W^-1) and written in the static_sim.json format, so pnr_refit.py and make_fig_pnr.py use them unchanged.

  --rounds 1  one ICA basis for all detections, diagonal GMM with --n components, Louvain into 5 groups
  --rounds 2  ICA basis 1 separates the 1-photon cluster (gap along its photon-number axis); the 1-photon
              detections get --n1 diagonal components in basis 1 (group C1), the rest a second ICA basis fitted
              without them and --n diagonal components, grouped into C2 .. C5+ by
                --grouping louvain  Louvain on the KL affinity, as build_pnr_model.py
                --grouping axis     the valleys of the histogram along the photon-number axis of basis 2 are
                                    the group boundaries; each component goes to the interval holding its mean.
                                    Photon numbers without a valley between them stay in one group.

usage: uv run python -m analysis.ica_gmm --dataset 10 --rounds 2 --n 20 --out FILE
"""

import argparse
import json

import numpy as np
import orjson
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture

from analysis.build_pnr_model import affinity, stable_communities
from analysis.datasets import decode_path, get
from analysis.make_fig_ica import fit_ica

CROP_PS = 250.0


def preamble_points(cfg):
    pts = np.asarray(orjson.loads(open(decode_path(cfg, cfg["model_db"]), "rb").read())["gm_data"]["counts"], float)
    keep = (np.abs(pts - np.median(pts, axis=0)) < CROP_PS).all(axis=1)
    return pts[keep]


class Basis:
    """ICA feature space z = (x - c) W, W with unit determinant (ps-like units), z[:, 0] the photon-number axis"""

    def __init__(self, x):
        self.c = x.mean(axis=0)
        self.W, _ = fit_ica(x - self.c)
        self.M = np.linalg.inv(self.W)

    def to_z(self, x):
        return (x - self.c) @ self.W

    def to_x(self, means_z, covs_z):
        return means_z @ self.M + self.c, np.einsum("ji,kjl,lm->kim", self.M, covs_z, self.M)


def diag_gmm(z, n, seed=42):
    gm = GaussianMixture(n_components=n, covariance_type="diag", random_state=seed).fit(z)
    return gm.means_, np.stack([np.diag(v) for v in gm.covariances_]), gm.weights_


def one_photon_split(z0):
    """threshold along the photon-number axis in the middle of the gap between the 1-photon cluster (largest z0)
    and the rest: the longest run of nearly empty bins above the median"""
    h, e = np.histogram(z0, bins=np.arange(np.percentile(z0, 0.1), np.percentile(z0, 99.9) + 1, 1.0))
    smooth = np.convolve(h, np.ones(5) / 5, mode="same")
    lo = int(np.searchsorted(e, np.median(z0)))
    empty = smooth[lo:] < 0.02 * smooth.max()
    best, start = (0, 0), None
    for i, v in enumerate(np.append(empty, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            best = max(best, (i - start, start))
            start = None
    length, s = best
    if length == 0:
        raise RuntimeError("no gap between the 1-photon cluster and the rest")
    return e[lo + s + length // 2]


def valley_boundaries(z0, prominence=0.05, bin_width=0.5):
    """minima between successive peaks of the smoothed histogram along the photon-number axis, high z0 first"""
    h, e = np.histogram(z0, bins=np.arange(np.percentile(z0, 0.1), np.percentile(z0, 99.9), bin_width))
    smooth = gaussian_filter1d(h.astype(float), 1.5)
    pk, _ = find_peaks(smooth, prominence=prominence * smooth.max())
    return sorted((e[a + int(np.argmin(smooth[a:b]))] for a, b in zip(pk[:-1], pk[1:])), reverse=True)


def groups_from(components, parts):
    means, covs, weights = components
    out = []
    for com in parts:
        idx = sorted(com)
        w = weights[idx] / weights[idx].sum()
        out.append(dict(center_of_mass=np.average(means[idx], weights=w, axis=0).tolist(),
                        gm_photon=dict(photon_covariances=covs[idx].tolist(), photon_means=means[idx].tolist(),
                                       photon_weights=w.tolist(), photon_num_components=len(idx)),
                        preamble_weight=float(weights[idx].sum())))
    return out


def build(cfg, rounds=2, n=20, n1=3, seed=42, grouping="louvain"):
    x = preamble_points(cfg)
    b1 = Basis(x)
    z = b1.to_z(x)
    res = np.arange(0.5, 3.01, 0.05)
    if rounds == 1:
        mz, cz, w = diag_gmm(z, n, seed)
        mx, cx = b1.to_x(mz, cz)
        parts, r, f = stable_communities(affinity(mx, cx), 5, res)
        groups = groups_from((mx, cx, w), parts)
        info = dict(resolution=r, partition_freq=f)
    else:
        one = z[:, 0] > one_photon_split(z[:, 0])
        frac1 = one.mean()
        m1, c1, w1 = diag_gmm(z[one], n1, seed)
        m1, c1 = b1.to_x(m1, c1)
        b2 = Basis(x[~one])
        z2 = b2.to_z(x[~one])
        mz, cz, w = diag_gmm(z2, n, seed)
        mx, cx = b2.to_x(mz, cz)
        ang = lambda W: np.degrees(np.arctan2(W[1, 0], W[0, 0])) % 180
        info = dict(one_photon_fraction=float(frac1), projection_angles_deg=[ang(b1.W), ang(b2.W)])
        if grouping == "louvain":
            parts, r, f = stable_communities(affinity(mx, cx), 4, res)
            info |= dict(resolution=r, partition_freq=f)
        else:
            edges = valley_boundaries(z2[:, 0])
            interval = np.searchsorted(-np.asarray(edges), -mz[:, 0])  # 0: above the highest valley (2 photons)
            parts = [set(np.flatnonzero(interval == i)) for i in range(len(edges) + 1)]
            parts = [p for p in parts if p]
            info |= dict(boundaries=[float(b) for b in edges], partition_freq=1.0)
        groups = groups_from((m1, c1, w1 * frac1), [set(range(n1))]) + groups_from((mx, cx, w * (1 - frac1)), parts)
    groups.sort(key=lambda g: g["center_of_mass"][1], reverse=True)
    return groups, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["10", "20"], required=True)
    ap.add_argument("--rounds", type=int, choices=[1, 2], default=2)
    ap.add_argument("--n", type=int, default=20, help="components (rounds 2: of the multi-photon part)")
    ap.add_argument("--n1", type=int, default=3, help="rounds 2: components of the 1-photon cluster")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grouping", choices=["louvain", "axis"], default="louvain")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    groups, info = build(get(a.dataset), a.rounds, a.n, a.n1, a.seed, a.grouping)
    with open(a.out, "w") as f:
        json.dump(groups, f)
    print(json.dumps(info), " ".join(str(g["gm_photon"]["photon_num_components"]) for g in groups))


if __name__ == "__main__":
    main()
