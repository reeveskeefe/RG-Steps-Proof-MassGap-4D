#!/usr/bin/env python3
# rp_histogram.py
# Histogram the RP test scores before/after one RG step (fast RP tester).

import os
import numpy as np
import matplotlib.pyplot as plt
from rg_step_prover import (
    LatticeSU3, rg_step, rp_test_scores_fast as rp_test_scores,
    heat_kernel_smooth, project_to_su3
)

OUT_DIR_FIGS = "figs"

def ensure_dir():
    os.makedirs(OUT_DIR_FIGS, exist_ok=True)

def ensemble(T, L, N, seed=21, K_smooth=1, tau_seed=0.6):
    rng = np.random.default_rng(seed)
    ens = []
    for _ in range(N):
        lat = LatticeSU3(T=T, L=L, rng=rng)
        for t in range(T):
            for x in range(L):
                for y in range(L):
                    for z in range(L):
                        for mu in range(4):
                            U = np.eye(3, dtype=complex)
                            for _k in range(K_smooth):
                                U = heat_kernel_smooth(U, tau=tau_seed, rng=rng)
                                U = project_to_su3(U)
                            lat.links[t, x, y, z, mu] = U
        ens.append(lat)
    return ens

def main():
    ensure_dir()
    T, L, N = 4, 4, 24
    b, bt, tau = 2, 2, 0.2
    rng = np.random.default_rng(9)

    ens = ensemble(T, L, N, seed=9, K_smooth=1, tau_seed=0.6)
    scores_before = []
    scores_after = []

    for lat in ens:
        sb = rp_test_scores(lat, n_obs=64, degree=3, rng=rng)
        scores_before.extend(list(sb))
        lat1 = rg_step(lat, b=b, b_t=bt, tau=tau, rng=rng)
        sa = rp_test_scores(lat1, n_obs=64, degree=3, rng=rng)
        scores_after.extend(list(sa))

    scores_before = np.array(scores_before, dtype=float)
    scores_after = np.array(scores_after, dtype=float)

    # BEFORE
    plt.figure()
    plt.hist(scores_before, bins=30)
    plt.xlabel("<F θF> (before RG)")
    plt.ylabel("count")
    plt.title("Reflection-Positivity Scores: BEFORE")
    fig1 = os.path.join(OUT_DIR_FIGS, "rp_hist_before.png")
    plt.savefig(fig1, dpi=160)
    plt.close()

    # AFTER
    plt.figure()
    plt.hist(scores_after, bins=30)
    plt.xlabel("<F θF> (after RG)")
    plt.ylabel("count")
    plt.title("Reflection-Positivity Scores: AFTER")
    fig2 = os.path.join(OUT_DIR_FIGS, "rp_hist_after.png")
    plt.savefig(fig2, dpi=160)
    plt.close()

    print("Saved:", fig1)
    print("Saved:", fig2)

if __name__ == "__main__":
    main()