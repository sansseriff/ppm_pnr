"""
Photon-number statistics without a Poisson prior, and without hand-tuned alignment offsets.

The photon-number model (inter/static_sim.json) is a set of groups C_j, j = 1..5+, each a fixed
mixture of 2D gaussians fit at a single attenuation. For every attenuation we fit only
  - the mixing weights pi_j of the groups (plus a weak uniform background for dark counts), and
  - one shared 2D shift v of the whole model (replaces offset_correction_*.json and its fudge factors)
by maximum likelihood (EM) on the measured (t_A, t_B) pairs, each referenced to its true PPM slot.

The fitted pi_j are an estimate of the detected photon-number distribution that does not assume
Poisson statistics, so comparing them with Poisson(mu) is a genuine test. For reference the script
also reports the legacy-style assignment, where the group weights are fixed to Poisson(mu, j).

usage: uv run python -m analysis.pnr_refit --dataset 20   (or 10)
"""

import argparse
import json
import math
import os

import numpy as np
import orjson
from scipy.special import logsumexp

from analysis.datasets import decode_path, get

MISSING = "D"
VALID = ("A", "B")  # correct / incorrect: a single tag inside the data region of the frame


def load_groups(path):
    groups = []
    for g in orjson.loads(open(path, "rb").read()):
        gp = g["gm_photon"]
        w = np.asarray(gp["photon_weights"], float)
        groups.append(
            dict(
                means=np.asarray(gp["photon_means"], float),
                covs=np.asarray(gp["photon_covariances"], float),
                weights=w / w.sum(),
            )
        )
    return groups


def load_events(path, T):
    d = orjson.loads(open(path, "rb").read())
    xy, n_missing, n_total = [], 0, 0
    for stream in d["results"]:
        for e in stream:
            n_total += 1
            if e["result"] == MISSING:
                n_missing += 1
            elif e["result"] in VALID:
                xy.append((e["tag_x"] - T * e["true"], e["tag_y"] - T * e["true"]))
    return np.asarray(xy), n_missing, n_total


def component_logpdf(x, means, covs):
    """(N, K) log N(x; mean_k, cov_k)"""
    inv = np.linalg.inv(covs)
    logdet = np.linalg.slogdet(covs)[1]
    d = x[:, None, :] - means[None]
    maha = np.einsum("nki,kij,nkj->nk", d, inv, d)
    return -0.5 * maha - 0.5 * logdet - np.log(2 * np.pi)


def flatten(groups):
    means = np.concatenate([g["means"] for g in groups])
    covs = np.concatenate([g["covs"] for g in groups])
    inner_w = np.concatenate([g["weights"] for g in groups])
    label = np.concatenate([np.full(len(g["weights"]), j) for j, g in enumerate(groups)])
    return means, covs, inner_w, label


def group_loglik(x, groups, v):
    """(N, J) log L_j(x) with the whole model shifted by v."""
    means, covs, inner_w, label = flatten(groups)
    lc = component_logpdf(x, means + v, covs) + np.log(inner_w)
    J = len(groups)
    return np.stack([logsumexp(lc[:, label == j], axis=1) for j in range(J)], axis=1)


def fit_weights_and_shift(x, groups, v0, bg_logdensity, n_iter=200, tol=1e-9, common_mode=True):
    """EM for group weights pi (plus background) and a shared shift v of all component means.
    common_mode: restrict v to s*(1,1). A clock/frame offset delays t_A and t_B equally, while
    t_B - t_A is set by photon number and a fixed cable delay, so only a common-mode shift is physical."""
    means, covs, inner_w, label = flatten(groups)
    inv = np.linalg.inv(covs)
    J = len(groups)
    pi = np.full(J + 1, 1.0 / (J + 1))
    v = np.asarray(v0, float)
    prev = -np.inf
    for _ in range(n_iter):
        lc = component_logpdf(x, means + v, covs) + np.log(inner_w) + np.log(np.maximum(pi[label], 1e-300))[None]
        lbg = np.full((len(x), 1), bg_logdensity + np.log(pi[J]))
        lall = np.concatenate([lc, lbg], axis=1)
        norm = logsumexp(lall, axis=1, keepdims=True)
        ll = norm.sum()
        r = np.exp(lall - norm)  # (N, K+1)
        rc = r[:, :-1]
        pi = np.array([rc[:, label == j].sum() for j in range(J)] + [r[:, -1].sum()]) / len(x)
        # closed-form M-step for a shift shared by all components
        A = np.einsum("nk,kij->ij", rc, inv)
        b = np.einsum("nk,kij,nkj->i", rc, inv, x[:, None, :] - means[None])
        if common_mode:
            u = np.ones(2)
            v = u * (u @ b) / (u @ A @ u)
        else:
            v = np.linalg.solve(A, b)
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    return pi, v, ll


def init_shift_grid(x, groups, center, bg_logdensity, half_width=250.0, step=10.0, n_sub=4000, seed=0,
                    common_mode=True):
    """coarse grid search for the model shift, using equal group weights and a data subsample."""
    rng = np.random.default_rng(seed)
    xs = x[rng.choice(len(x), min(n_sub, len(x)), replace=False)]
    J = len(groups)
    grid = np.arange(-half_width, half_width + step / 2, step)
    pairs = [(d, d) for d in grid] if common_mode else [(dx, dy) for dx in grid for dy in grid]
    best, best_v = -np.inf, None
    for dx, dy in pairs:
        v = center + np.array([dx, dy])
        lg = group_loglik(xs, groups, v) - np.log(J)
        l = np.logaddexp(logsumexp(lg, axis=1), bg_logdensity).sum()
        if l > best:
            best, best_v = l, v
    return best_v


def poisson_pmf(mu, J):
    p = [math.exp(-mu) * mu**k / math.factorial(k) for k in range(J)]
    return np.array(p + [1 - sum(p)])  # last entry is J+ (k >= J)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["10", "20"], required=True)
    ap.add_argument("--model", default=None, help="default: <dataset>/inter/static_sim.json")
    ap.add_argument("--db", nargs="*", type=float, default=None, help="default: all attenuations")
    ap.add_argument("--free-shift", action="store_true", help="allow a general 2D shift (not physical)")
    ap.add_argument("--out", default=None, help="default: <dataset>/inter/pnr_refit.json")
    args = ap.parse_args()
    cfg = get(args.dataset)

    groups = load_groups(args.model or cfg["model"])
    J = len(groups)
    out = {}
    for db in sorted(args.db or cfg["dbs"], reverse=True):
        x, n_missing, n_total = load_events(decode_path(cfg, db), cfg["slot_ps"])
        mu_vac = -math.log(n_missing / n_total)
        # background: uniform over the bounding box of the data
        span = np.ptp(x, axis=0)
        bg = -np.log(span[0] * span[1])
        # initial shift: align the data median with the model's overall centroid, then refine on a grid
        means, _, inner_w, label = flatten(groups)
        center = np.median(x, axis=0) - np.average(means, weights=inner_w, axis=0)
        if not args.free_shift:
            center = np.full(2, center.mean())
        v0 = init_shift_grid(x, groups, center, bg, half_width=600.0, step=5.0, common_mode=not args.free_shift)
        pi, v, ll = fit_weights_and_shift(x, groups, v0, bg, common_mode=not args.free_shift)

        # detected photon-number distribution (conditional on >= 1 photon), from the fit
        p_det = pi[:J] / pi[:J].sum()
        n_det = len(x) * pi[:J].sum()
        # full distribution including vacuum, normalised to all frames
        measured = np.concatenate([[n_missing], p_det * n_det]) / (n_missing + n_det)

        # mu from the photon-number distribution alone (ML fit of a zero-truncated Poisson)
        mus = np.linspace(0.01, 40, 8000)
        k = np.arange(1, J + 1)
        def trunc_ll(mu):
            p = poisson_pmf(mu, J)[1:]
            return np.sum(p_det * np.log(p / p.sum() + 1e-300))
        mu_pnr = float(mus[np.argmax([trunc_ll(m) for m in mus])])

        # legacy-style hard assignment with Poisson(mu_vac) group weights, for comparison
        lg = group_loglik(x, groups, v)
        prior = poisson_pmf(mu_vac, J)[1:]
        hard_prior = np.bincount(np.argmax(lg + np.log(prior + 1e-300), axis=1), minlength=J)
        hard_free = np.bincount(np.argmax(lg + np.log(pi[:J] + 1e-300), axis=1), minlength=J)

        out[f"{db:.1f}"] = dict(
            n_total=n_total,
            n_missing=n_missing,
            n_valid=len(x),
            mu_vacuum=mu_vac,
            mu_pnr=mu_pnr,
            pi=pi.tolist(),
            shift=v.tolist(),
            loglik=ll,
            measured_with_vacuum=measured.tolist(),
            poisson_mu_vacuum=poisson_pmf(mu_vac, J).tolist(),
            hard_counts_free_weights=hard_free.tolist(),
            hard_counts_poisson_prior=hard_prior.tolist(),
        )
        print(
            f"{db:4.0f} dB  mu_vac={mu_vac:6.3f}  mu_pnr={mu_pnr:6.3f}  "
            f"shift=({v[0]:+7.2f},{v[1]:+7.2f})  bg={pi[J]:.4f}  "
            f"p_det(1..{J}+)={np.round(p_det, 3).tolist()}",
            flush=True,
        )

    with open(args.out or cfg["refit"], "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
