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
  --sigma-max  cap on the component width along the photon-number axis, so that components cannot straddle
               two photon-number peaks (EM continued from scikit-learn's fit with the clipped variance)

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

    def __init__(self, x=None, c=None, W=None):
        if x is not None:
            c = x.mean(axis=0)
            W, _ = fit_ica(x - c)
        self.c, self.W = np.asarray(c, float), np.asarray(W, float)
        self.M = np.linalg.inv(self.W)

    def to_z(self, x):
        return (x - self.c) @ self.W

    def to_x(self, means_z, covs_z):
        return means_z @ self.M + self.c, np.einsum("ji,kjl,lm->kim", self.M, covs_z, self.M)


def diag_gmm(z, n, seed=42, sigma_max=None, n_iter=300, tol=1e-7):
    """diagonal GMM in feature space. sigma_max caps each component's standard deviation along the photon-number
    axis (z[:, 0]): starting from scikit-learn's fit, EM is continued with the M-step variance along that axis
    clipped to sigma_max^2, so no component can spread across neighbouring photon-number peaks."""
    gm = GaussianMixture(n_components=n, covariance_type="diag", random_state=seed).fit(z)
    m, var, w = gm.means_.copy(), gm.covariances_.copy(), gm.weights_.copy()
    if sigma_max is not None:
        var[:, 0] = np.minimum(var[:, 0], sigma_max**2)
        prev = -np.inf
        for _ in range(n_iter):
            lp = (-0.5 * (((z[:, None, :] - m[None]) ** 2) / var[None]).sum(axis=2)
                  - 0.5 * np.log(var).sum(axis=1)[None] + np.log(w)[None])
            top = lp.max(axis=1, keepdims=True)
            ll_n = top[:, 0] + np.log(np.exp(lp - top).sum(axis=1))
            r = np.exp(lp - ll_n[:, None])
            nk = r.sum(axis=0) + 1e-12
            w = nk / len(z)
            m = (r.T @ z) / nk[:, None]
            var = (r.T @ z**2) / nk[:, None] - m**2 + 1e-6
            var[:, 0] = np.minimum(var[:, 0], sigma_max**2)
            ll = ll_n.mean()
            if ll - prev < tol:
                break
            prev = ll
    return m, np.stack([np.diag(v) for v in var]), w


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


def save_geometry(path, info, n, n1, sigma_max):
    """the ICA bases, the |1> cut and the valley boundaries of a rounds-2 axis-grouped model (next to the model)"""
    g = info["_geometry"]
    json.dump(dict(basis1=dict(c=g["basis1"].c.tolist(), W=g["basis1"].W.tolist()),
                   basis2=dict(c=g["basis2"].c.tolist(), W=g["basis2"].W.tolist()),
                   cut=float(g["cut"]), boundaries=info["boundaries"], n=n, n1=n1, sigma_max=sigma_max),
              open(path, "w"), indent=1)


def geometry_path(model_path):
    return str(model_path).replace(".json", "_geometry.json")


def local_refit(x, geometry, base_groups, seed=42, min_points=200):
    """refit the components of a rounds-2 axis-grouped model to other detections x (in the model's time frame),
    keeping its ICA bases, |1> cut and valley boundaries: the photon-number groups keep their meaning while the
    component shapes follow the data (e.g. the preamble of another attenuation). A group whose interval holds fewer
    than min_points detections, or receives no component, keeps its components from base_groups (the model, in the
    static_sim.json format). Returns groups in the same format, ordered by photon number."""
    b1 = Basis(c=geometry["basis1"]["c"], W=geometry["basis1"]["W"])
    b2 = Basis(c=geometry["basis2"]["c"], W=geometry["basis2"]["W"])
    x = x[(np.abs(x - b1.c) < 2 * CROP_PS).all(axis=1)]
    one = b1.to_z(x)[:, 0] > geometry["cut"]
    edges = geometry["boundaries"]
    J = len(edges) + 2
    out = [None] * J
    if one.sum() >= min_points:
        m1, c1, w1 = diag_gmm(b1.to_z(x[one]), geometry["n1"], seed)
        m1, c1 = b1.to_x(m1, c1)
        out[0] = groups_from((m1, c1, w1 * one.mean()), [set(range(geometry["n1"]))])[0]
    z2 = b2.to_z(x[~one])
    n = int(min(geometry["n"], (~one).sum() // min_points))  # fewer components for sparse data
    if n >= 1:
        mz, cz, w = diag_gmm(z2, n, seed, geometry["sigma_max"])
        mx, cx = b2.to_x(mz, cz)
        interval = np.searchsorted(-np.asarray(edges), -mz[:, 0])
        counts = np.bincount(np.searchsorted(-np.asarray(edges), -z2[:, 0]), minlength=J - 1)
        for i in range(J - 1):
            idx = set(np.flatnonzero(interval == i))
            if idx and counts[i] >= min_points:
                out[i + 1] = groups_from((mx, cx, w * (1 - one.mean())), [idx])[0]
    base = sorted(base_groups, key=lambda g: g["center_of_mass"][1], reverse=True)
    return [o if o is not None else base[j] for j, o in enumerate(out)]


def build(cfg, rounds=2, n=20, n1=3, seed=42, grouping="louvain", sigma_max=None):
    x = preamble_points(cfg)
    b1 = Basis(x)
    z = b1.to_z(x)
    res = np.arange(0.5, 3.01, 0.05)
    if rounds == 1:
        mz, cz, w = diag_gmm(z, n, seed, sigma_max)
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
        mz, cz, w = diag_gmm(z2, n, seed, sigma_max)
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
        info["_geometry"] = dict(basis1=b1, basis2=b2, cut=one_photon_split(z[:, 0]), x=x, one=one)
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
    ap.add_argument("--sigma-max", type=float, default=None,
                    help="cap on each component's standard deviation along the photon-number axis (ICA units)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    groups, info = build(get(a.dataset), a.rounds, a.n, a.n1, a.seed, a.grouping, a.sigma_max)
    with open(a.out, "w") as f:
        json.dump(groups, f)
    if a.rounds == 2 and a.grouping == "axis":
        save_geometry(geometry_path(a.out), info, a.n, a.n1, a.sigma_max)
    print(json.dumps({k: v for k, v in info.items() if not k.startswith("_")}), " ".join(str(g["gm_photon"]["photon_num_components"]) for g in groups))


if __name__ == "__main__":
    main()
