"""
Build the photon-number model: groups C_j of gaussian components, one group per photon number.

Steps (as in ppm_20ghz bi_gauss_community_detection.ipynb, made deterministic):
  1. take a GMM of the preamble (slot-0) response at a moderate mean photon number, either one stored
     in the decode file (--gm-idx) or refit to the stored preamble points (--refit N)
  2. drop far-field components (> --far-cutoff ps from the weighted centre) and degenerate ones
     (axis ratio > 10, sum of singular values outside [10, 2500], weight < 0.005)
  3. symmetric KL divergence between all component pairs (positions /100, covariances /1e4),
     affinity max(0, 40 - D) normalised to 1, Louvain communities. Louvain is randomised, so the
     lowest resolution that yields --n-groups groups is used, taking the modal partition over seeds
  4. order groups by the t_B of their centre of mass, highest first (1-photon detections have the
     latest high-threshold crossing), and write them in the static_sim.json format

The published models are
  10 GHz: ppm_10ghz/inter/static_sim.json = this script, --dataset 10 (35-component refit of the 28 dB preamble)
  20 GHz: ppm_20ghz/inter/static_sim.json = ppm_20ghz/deprecated/bi_gauss_community_detection.ipynb (Nov 2023,
          gm_list[15] of the 24 dB file, unseeded Louvain). `--dataset 20` rebuilds it deterministically from the
          same 35-component fit and reproduces the published grouping (same partition in 90% of seeds).
  10 GHz rebuild with the locked environment matches the published file to ~1e-11 relative.
The refit uses sklearn's GaussianMixture, whose result can change slightly between scikit-learn versions.

usage: uv run python -m analysis.build_pnr_model --dataset 10 [--out ppm_10ghz/inter/static_sim_rebuilt.json]
       uv run python -m analysis.build_pnr_model --decode FILE (--gm-idx K | --refit N) --out FILE
"""

import argparse
import json

import networkx as nx
import numpy as np
import orjson
import scipy.linalg
from sklearn.mixture import GaussianMixture

from analysis.datasets import decode_path, get

# default model source per dataset
DEFAULTS = {"10": dict(refit=35, gm_idx=None), "20": dict(refit=None, gm_idx=15)}


def kl_mvn(to, fr):
    """KL(to || fr) for gaussians given as (mean, cov)"""
    m_to, S_to = to
    m_fr, S_fr = fr
    d = m_fr - m_to
    c = scipy.linalg.cho_factor(S_fr)
    solve = lambda B: scipy.linalg.cho_solve(c, B)
    logdet = lambda S: np.linalg.slogdet(S)[1]
    return (np.trace(solve(S_to)) + logdet(S_fr) - logdet(S_to) + d @ solve(d) - len(d)) / 2.0


def filter_components(means, covs, weights, far_cutoff):
    center = np.average(means, weights=weights, axis=0)
    keep = np.linalg.norm(means - center, axis=1) <= far_cutoff
    means, covs, weights = means[keep], covs[keep], weights[keep] / weights[keep].sum()
    sv = np.linalg.svd(covs, compute_uv=False)
    sig = sv.sum(axis=1)
    keep = (sv[:, 0] / sv[:, 1] <= 10) & (sig <= 2500) & (sig >= 10) & (weights > 0.005)
    return means[keep], covs[keep], weights[keep] / weights[keep].sum()


def affinity(means, covs, far_field=40.0):
    n = len(means)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                a = (means[i] / 100, covs[i] / 1e4)
                b = (means[j] / 100, covs[j] / 1e4)
                D[i, j] = kl_mvn(b, a) + kl_mvn(a, b)
    A = np.clip(far_field - D, 0, None)
    A = A / A.max()
    np.fill_diagonal(A, 0)
    return A


def stable_communities(A, n_groups, resolutions, n_seeds=50):
    """Louvain is randomised; scan seeds (and resolutions, lowest first) and return the most frequent
    partition with exactly n_groups communities, with its frequency among the runs at that resolution."""
    G = nx.from_numpy_array(A)
    for res in resolutions:
        counts = {}
        for seed in range(n_seeds):
            part = nx.community.louvain_communities(G, resolution=res, seed=seed)
            if len(part) == n_groups:
                key = frozenset(frozenset(c) for c in part)
                counts[key] = counts.get(key, 0) + 1
        if counts:
            best = max(counts, key=counts.get)
            return [set(c) for c in best], res, counts[best] / n_seeds
    raise RuntimeError(f"no partition with {n_groups} groups found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["10", "20"], default=None,
                    help="use the dataset's model attenuation and fit (see DEFAULTS)")
    ap.add_argument("--decode", default=None)
    ap.add_argument("--gm-idx", type=int, default=None)
    ap.add_argument("--refit", type=int, default=None, help="refit a GMM with this many components")
    ap.add_argument("--far-cutoff", type=float, default=1000.0)
    ap.add_argument("--n-groups", type=int, default=5, help="photon-number groups: 1..n-1 and n+")
    ap.add_argument("--out", default=None, help="default with --dataset: <dataset>/inter/static_sim_rebuilt.json")
    args = ap.parse_args()
    if args.dataset:
        cfg = get(args.dataset)
        args.decode = args.decode or str(decode_path(cfg, cfg["model_db"]))
        if args.gm_idx is None and args.refit is None:
            args.gm_idx, args.refit = DEFAULTS[args.dataset]["gm_idx"], DEFAULTS[args.dataset]["refit"]
        args.out = args.out or str(cfg["inter"] / "static_sim_rebuilt.json")
    if not args.decode or not args.out or (args.gm_idx is None and not args.refit):
        ap.error("give --dataset, or --decode, --out and one of --gm-idx / --refit")

    d = orjson.loads(open(args.decode, "rb").read())
    if args.refit:
        pts = np.asarray(d["gm_data"]["counts"], float)
        gm = GaussianMixture(n_components=args.refit, random_state=42).fit(pts)
        means, covs, weights = gm.means_, gm.covariances_, gm.weights_
    else:
        g = d["gm_data"]["gm_list"][args.gm_idx]
        means, covs, weights = (np.asarray(g[k], float) for k in ("means", "covariances", "weights"))
    n_in = len(weights)
    means, covs, weights = filter_components(means, covs, weights, args.far_cutoff)

    parts, res, freq = stable_communities(affinity(means, covs), args.n_groups, np.arange(0.5, 2.01, 0.05))
    groups = []
    for com in parts:
        idx = sorted(com)
        w = weights[idx] / weights[idx].sum()
        groups.append(
            dict(
                center_of_mass=np.average(means[idx], weights=w, axis=0).tolist(),
                gm_photon=dict(
                    photon_covariances=covs[idx].tolist(),
                    photon_means=means[idx].tolist(),
                    photon_weights=w.tolist(),
                    photon_num_components=len(idx),
                ),
                preamble_weight=float(weights[idx].sum()),
            )
        )
    groups.sort(key=lambda g: g["center_of_mass"][1], reverse=True)
    with open(args.out, "w") as f:
        json.dump(groups, f)
    print(f"{n_in} components -> {len(weights)} kept -> {len(groups)} groups "
          f"(Louvain resolution {res:.2f}, partition found in {100 * freq:.0f}% of seeds)")
    for j, g in enumerate(groups):
        print(f"  C{j + 1}: {g['gm_photon']['photon_num_components']:2d} comps, "
              f"com=({g['center_of_mass'][0]:7.1f}, {g['center_of_mass'][1]:7.1f}), "
              f"preamble weight {g['preamble_weight']:.3f}")


if __name__ == "__main__":
    main()
