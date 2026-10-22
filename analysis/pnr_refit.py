"""
Photon-number attribution with the photon-number model, its cluster weights and timing offset fitted.

The photon-number model (inter/static_sim.json) is a set of groups C_j, j = 1..5+, each a fixed mixture of 2D
gaussians fit at one attenuation (the cluster shapes). For every attenuation the relative weights ("heights")
of the five groups, a weak uniform background, and one timing offset (a shift applied equally to t_A and t_B)
are fitted by maximum likelihood (expectation-maximization). No Poisson statistics are assumed, so comparing the
resulting photon-number fractions with Poisson(mu) (Fig. 7) is a test. The offset replaces the hand-derived
offset_correction_*.json vectors and the in-sample max_offsets.json of the 2023 notebooks. Each detection is
then assigned the most probable (slot, photon number) among the three slots nearest the slope estimate, as in
the 2023 pnr_caller.

Robustness check (notebooks/photon_number_model.ipynb, Supplementary Note 6): the 2023 weighting, group weights
fixed to Poisson(mu, j) at the measured mu (make_fig_decoding.py) with only the offset fitted.

Needs <dataset>/figs/ppm_decoding_<tag>_table.json (make_fig_decoding.py) for mu.

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


def fit_weights_and_shift(x, groups, v0, bg_logdensity, n_iter=200, tol=1e-9, common_mode=True, fixed_weights=None):
    """EM for group weights pi (plus background) and a shared shift v of all component means.
    fixed_weights: relative group weights to keep fixed (e.g. Poisson(mu, j)); only the background fraction
    and the shift are then fitted.
    common_mode: restrict v to s*(1,1). A clock/frame offset delays t_A and t_B equally, while
    t_B - t_A is set by photon number and a fixed cable delay, so only a common-mode shift is physical."""
    means, covs, inner_w, label = flatten(groups)
    inv = np.linalg.inv(covs)
    J = len(groups)
    pi = np.full(J + 1, 1.0 / (J + 1))
    if fixed_weights is not None:
        fixed = np.asarray(fixed_weights, float) / np.sum(fixed_weights)
        pi = np.append(0.99 * fixed, 0.01)
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
        if fixed_weights is None:
            pi = np.array([rc[:, label == j].sum() for j in range(J)] + [r[:, -1].sum()]) / len(x)
        else:
            f_bg = r[:, -1].sum() / len(x)
            pi = np.append((1 - f_bg) * fixed, f_bg)
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


def load_stream_events(path, T):
    """valid detections with their repetition, slope-estimated slot and true slot"""
    d = orjson.loads(open(path, "rb").read())
    rows = [(si, e["tag_x"], e["tag_y"], e["true"], e["measured"])
            for si, stream in enumerate(d["results"]) for e in stream if e["result"] in VALID]
    a = np.array(rows, float)
    return dict(stream=a[:, 0].astype(int), xy=a[:, 1:3], true=a[:, 3].astype(int), measured=a[:, 4].astype(int))


def joint_attribution(ev, groups, weights, v, T):
    """most probable (slot, photon number) over the slots slope-estimate + {-1, 0, 1} (2023 pnr_caller)"""
    J = len(groups)
    best = np.full(len(ev["true"]), -np.inf)
    slot = np.zeros(len(ev["true"]), int)
    num = np.zeros(len(ev["true"]), int)
    for k in (-1, 0, 1):
        s = ev["measured"] + k
        lg = group_loglik(ev["xy"] - T * s[:, None], groups, v) + np.log(np.maximum(weights, 1e-300))
        j = np.argmax(lg, axis=1)
        sc = lg[np.arange(len(j)), j]
        upd = sc > best
        best[upd], slot[upd], num[upd] = sc[upd], s[upd], j[upd]
    return slot, num


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["10", "20"], required=True)
    ap.add_argument("--model", default=None, help="default: <dataset>/inter/static_sim.json")
    ap.add_argument("--db", nargs="*", type=float, default=None, help="default: all attenuations")
    ap.add_argument("--out", default=None, help="default: <dataset>/inter/pnr_refit.json")
    ap.add_argument("--local", action="store_true",
                    help="per attenuation, refit the model's components to that attenuation's preamble in the model's "
                         "ICA bases first (as the decoder does; fair_comparison.calibrate_structured)")
    args = ap.parse_args()
    cfg = get(args.dataset)
    T = cfg["slot_ps"]
    mu_table = {r["db"]: r["mu"] for r in json.loads(open(cfg["table"]).read())["rows"]}

    model = args.model or cfg["model"]
    groups = load_groups(model)
    J = len(groups)
    out = {}
    for db in sorted(args.db or cfg["dbs"], reverse=True):
        x, n_missing, n_total = load_events(decode_path(cfg, db), T)
        if args.local:
            from analysis.fair_comparison import calibrate_structured
            pre = np.asarray(orjson.loads(open(decode_path(cfg, db), "rb").read())["gm_data"]["counts"], float)
            groups = calibrate_structured(pre, model, local=True)[0]
        mu = mu_table[int(db)]
        prior = poisson_pmf(mu, J)[1:]
        span = np.ptp(x, axis=0)
        bg = -np.log(span[0] * span[1])  # background: uniform over the bounding box of the data
        means, _, inner_w, label = flatten(groups)
        center = np.full(2, (np.median(x, axis=0) - np.average(means, weights=inner_w, axis=0)).mean())
        v0 = init_shift_grid(x, groups, center, bg, half_width=600.0, step=5.0)
        # cluster weights + background + common-mode timing offset, fitted by EM (no Poisson prior)
        pi, v, ll = fit_weights_and_shift(x, groups, v0, bg)
        weights = pi[:J] / pi[:J].sum()

        # joint slot + photon-number attribution of every detection (Figs. 6, 7)
        ev = load_stream_events(decode_path(cfg, db), T)
        slot, num = joint_attribution(ev, groups, weights, v, T)
        counts = np.bincount(num, minlength=J)
        measured = np.concatenate([[n_missing], counts]) / (n_missing + len(num))
        slot_accuracy = float(np.mean(slot == ev["true"]))
        # mu implied by the fitted weights alone (zero-truncated Poisson), independent of the vacuum fraction
        mus = np.linspace(0.01, 40, 8000)
        trunc_ll = [np.sum(weights * np.log(poisson_pmf(m, J)[1:] / poisson_pmf(m, J)[1:].sum() + 1e-300)) for m in mus]

        # robustness check: the 2023 weighting, Poisson(mu) at the measured mu, with only the offset fitted
        pi_p, v_p, _ = fit_weights_and_shift(x, groups, v, bg, fixed_weights=prior)
        slot_p, num_p = joint_attribution(ev, groups, prior, v_p, T)

        out[f"{db:.1f}"] = dict(
            n_total=n_total, n_missing=n_missing, n_valid=len(x), mu=mu,
            mu_vacuum=-math.log(n_missing / n_total), mu_from_weights=float(mus[int(np.argmax(trunc_ll))]),
            weights=weights.tolist(), background_fraction=float(pi[J]), shift=v.tolist(), loglik=ll,
            measured_with_vacuum=measured.tolist(), counts=counts.tolist(), slot_accuracy=slot_accuracy,
            poisson=poisson_pmf(mu, J).tolist(),
            poisson_weighted=dict(shift=v_p.tolist(), counts=np.bincount(num_p, minlength=J).tolist(),
                                  slot_accuracy=float(np.mean(slot_p == ev["true"]))),
        )
        print(f"{db:4.0f} dB  mu={mu:6.3f}  shift={v[0]:+7.2f} ps  slot acc {slot_accuracy:.4f}  "
              f"fractions(1..{J}+)={np.round(counts / counts.sum(), 3).tolist()}", flush=True)

    with open(args.out or cfg["refit"], "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
