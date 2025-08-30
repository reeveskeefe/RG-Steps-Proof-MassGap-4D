#!/usr/bin/env python3
# rg_param_scan.py
# Grid-scan over (b_space, b_time, tau) to map contraction strength A (fast path).

import os, csv
import numpy as np
import matplotlib.pyplot as plt
from rg_step_prover import (
    LatticeSU3, rg_step, kp_norm_two_point_fast,
    heat_kernel_smooth, project_to_su3
)

OUT_DIR_RESULTS = "results"
OUT_DIR_FIGS = "figs"

def ensure_dirs():
    os.makedirs(OUT_DIR_RESULTS, exist_ok=True)
    os.makedirs(OUT_DIR_FIGS, exist_ok=True)

def ensemble(T, L, N, seed=5, K_smooth=1, tau_seed=0.6):
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

def eta_of_lat(lat, alpha=0.6, gamma=0.6, rmax=2):
    P = lat.plaquette_scalar()
    return kp_norm_two_point_fast(P - P.mean(), lat.T, lat.L, alpha, gamma, rmax)

def one_A_for_params(ens, b_space, b_time, tau, alpha=0.6, gamma=0.6, rmax=2, rng=None):
    rng = rng or np.random.default_rng(0)
    eta0 = np.array([eta_of_lat(lat, alpha, gamma, rmax) for lat in ens], dtype=float)
    ens1 = [rg_step(lat, b=b_space, b_t=b_time, tau=tau, rng=rng) for lat in ens]
    eta1 = np.array([eta_of_lat(lat, alpha, gamma, rmax) for lat in ens1], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = eta1 / (eta0 ** 2 + 1e-16)
    ratios = ratios[np.isfinite(ratios)]
    return float(np.mean(ratios)), float(np.std(ratios)), float(np.mean(eta0)), float(np.mean(eta1))

def main():
    ensure_dirs()

    T, L = 4, 4
    N_cfg = 24
    taus = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    b_spaces = [2, 3]
    b_times = [1, 2, 3]
    alpha, gamma, rmax = 0.6, 0.6, 2

    rng = np.random.default_rng(5)
    base_ens = ensemble(T, L, N_cfg, seed=5, K_smooth=1, tau_seed=0.6)

    csv_path = os.path.join(OUT_DIR_RESULTS, "param_scan_A.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["b_space", "b_time", "tau", "A_mean", "A_std", "eta0_mean", "eta1_mean"])
        for b in b_spaces:
            for bt in b_times:
                A_curve = []
                for tau in taus:
                    A_mean, A_std, e0, e1 = one_A_for_params(
                        [lat.copy() for lat in base_ens], b, bt, tau, alpha, gamma, rmax, rng
                    )
                    w.writerow([b, bt, tau, A_mean, A_std, e0, e1])
                    A_curve.append((tau, A_mean))
                # plot A vs tau for this (b, bt)
                xs = [p[0] for p in A_curve]
                ys = [p[1] for p in A_curve]
                plt.figure()
                plt.plot(xs, ys, marker="o")
                plt.xlabel("tau (heat-kernel smoothing)")
                plt.ylabel("A estimate (mean)")
                plt.title(f"A vs tau  |  b_space={b}, b_time={bt}")
                figpath = os.path.join(OUT_DIR_FIGS, f"A_vs_tau_b{b}_bt{bt}.png")
                plt.savefig(figpath, dpi=160)
                plt.close()

    print("Saved:", csv_path)
    print("Saved plots to:", OUT_DIR_FIGS)

if __name__ == "__main__":
    main()