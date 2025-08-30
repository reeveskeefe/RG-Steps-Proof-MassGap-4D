#!/usr/bin/env python3
# rg_multi_step_experiment.py
# Multi-step RG run (fast path): track eta_k contraction and A estimates, with
# adaptive blocking factors so steps keep running even when T or L become 1.

import os, csv
import numpy as np
import matplotlib.pyplot as plt
from rg_step_prover import (
    LatticeSU3, rg_step, kp_norm_two_point_fast, rp_test_scores_fast,
    heat_kernel_smooth, project_to_su3
)

OUT_DIR_RESULTS = "results"
OUT_DIR_FIGS = "figs"

def ensure_dirs():
    os.makedirs(OUT_DIR_RESULTS, exist_ok=True)
    os.makedirs(OUT_DIR_FIGS, exist_ok=True)

def ensemble(T, L, N, seed=11, K_smooth=1, tau_seed=0.6):
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

def main():
    ensure_dirs()

    # Parameters (feel free to tweak)
    T, L = 4, 4
    N_cfg = 36
    steps = 4
    b_space_pref, b_time_pref = 2, 2   # preferred blocking factors
    tau = 0.2
    alpha, gamma, rmax = 0.6, 0.6, 2
    K_smooth = 1
    rp_nobs = 32

    rng = np.random.default_rng(11)
    ens = ensemble(T, L, N_cfg, seed=11, K_smooth=K_smooth, tau_seed=0.6)

    eta_means, eta_stds = [], []
    A_means, A_stds = [], []

    # Track lattice sizes and effective factors for the CSV log
    dims_log = []        # (k, T_k, L_k)
    eff_log = []         # (k, b_eff, bt_eff)

    # Step 0 (fine lattice)
    etas = np.array([eta_of_lat(lat, alpha, gamma, rmax) for lat in ens], dtype=float)
    eta_means.append(float(np.mean(etas)))
    eta_stds.append(float(np.std(etas)))
    dims_log.append((0, ens[0].T, ens[0].L))
    prev_etas = etas

    # Multi-step loop with ADAPTIVE blocking
    for k in range(1, steps + 1):
        Tk, Lk = ens[0].T, ens[0].L
        # Choose effective factors that divide current sizes (or skip with 1)
        b_eff  = b_space_pref if (Lk % b_space_pref == 0 and Lk // b_space_pref >= 1) else 1
        bt_eff = b_time_pref  if (Tk % b_time_pref == 0 and Tk // b_time_pref >= 1) else 1

        print(f"[step {k}] lattice {Tk}x{Lk}  →  b_space_eff={b_eff}, b_time_eff={bt_eff}")

        ens = [rg_step(lat, b=b_eff, b_t=bt_eff, tau=tau, rng=rng) for lat in ens]

        etas = np.array([eta_of_lat(lat, alpha, gamma, rmax) for lat in ens], dtype=float)
        eta_means.append(float(np.mean(etas)))
        eta_stds.append(float(np.std(etas)))

        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = etas / (prev_etas ** 2 + 1e-16)
        ratios = ratios[np.isfinite(ratios)]
        A_means.append(float(np.mean(ratios)) if ratios.size else float("nan"))
        A_stds.append(float(np.std(ratios)) if ratios.size else float("nan"))

        dims_log.append((k, ens[0].T, ens[0].L))
        eff_log.append((k, b_eff, bt_eff))
        prev_etas = etas

    # Save CSV (includes dims and effective factors)
    csv_path = os.path.join(OUT_DIR_RESULTS, "multi_step_eta_A.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "T_k", "L_k", "eta_mean", "eta_std", "A_mean(k->k+1)", "A_std(k->k+1)", "b_eff", "bt_eff"])
        for k in range(len(eta_means)):
            T_k, L_k = dims_log[k][1], dims_log[k][2]
            if k == 0:
                w.writerow([k, T_k, L_k, eta_means[k], eta_stds[k], "", "", "", ""])
            else:
                b_eff = eff_log[k-1][1]
                bt_eff = eff_log[k-1][2]
                w.writerow([k, T_k, L_k, eta_means[k], eta_stds[k], A_means[k-1], A_stds[k-1], b_eff, bt_eff])

    # Plot eta_k vs k
    ks = np.arange(len(eta_means))
    plt.figure()
    plt.plot(ks, eta_means, marker="o")
    plt.yscale("log")
    plt.xlabel("RG step k")
    plt.ylabel("eta_k (mean across ensemble)")
    plt.title("Quadratic Contraction: eta_k vs k")
    fig1 = os.path.join(OUT_DIR_FIGS, "eta_vs_k.png")
    plt.savefig(fig1, dpi=160)
    plt.close()

    # Plot A_k vs k
    if A_means:
        kk = np.arange(1, len(eta_means))
        plt.figure()
        plt.plot(kk, A_means, marker="o")
        plt.xlabel("RG step (k→k+1)")
        plt.ylabel("A estimate (mean)")
        plt.title("Estimated A from eta_{k+1}/eta_k^2")
        fig2 = os.path.join(OUT_DIR_FIGS, "A_vs_k.png")
        plt.savefig(fig2, dpi=160)
        plt.close()

    print("Saved:", csv_path)
    print("Saved:", fig1)
    if A_means:
        print("Saved:", fig2)

if __name__ == "__main__":
    main()