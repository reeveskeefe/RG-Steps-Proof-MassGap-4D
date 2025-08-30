#!/usr/bin/env python3
# locality_decay.py
# Check finite-range locality by plotting average |C2| vs lattice distance d (before/after one RG step).

import os
import numpy as np
import matplotlib.pyplot as plt
from rg_step_prover import (
    LatticeSU3, rg_step, heat_kernel_smooth, project_to_su3
)

OUT_DIR_FIGS = "figs"

def ensure_dir():
    os.makedirs(OUT_DIR_FIGS, exist_ok=True)

def ensemble(T, L, N, seed=13, K_smooth=1, tau_seed=0.6):
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

def l1_periodic(a, b, Ls):
    d = 0
    for i in range(4):
        L = Ls[i]
        da = abs(int(a[i]) - int(b[i]))
        d += min(da, L - da)
    return d

def plaquette_coords(lat):
    dirs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    coords = []
    for t in range(lat.T):
        for x in range(lat.L):
            for y in range(lat.L):
                for z in range(lat.L):
                    for idx,_ in enumerate(dirs):
                        coords.append((t,x,y,z,idx))
    return np.array(coords, dtype=int)

def avg_abs_C2_vs_distance(lat, rmax=5):
    coords = plaquette_coords(lat)  # (t,x,y,z,or)
    P = lat.plaquette_scalar()
    v = P - P.mean()
    T, L = lat.T, lat.L
    Ls = (T, L, L, L)
    sums, counts = {}, {}
    N = len(v)
    for i in range(N):
        ai = coords[i]
        for j in range(N):
            if i == j:
                continue
            aj = coords[j]
            d = l1_periodic(ai, aj, Ls)
            if d > rmax:
                continue
            c2 = (v[i] * v[j]).real
            sums[d] = sums.get(d, 0.0) + abs(c2)
            counts[d] = counts.get(d, 0) + 1
    ds = sorted(sums.keys())
    ys = [sums[d] / max(1, counts[d]) for d in ds]
    return np.array(ds, dtype=float), np.array(ys, dtype=float)

def main():
    ensure_dir()
    T, L, N = 4, 4, 24
    b, bt, tau = 2, 2, 0.2
    rng = np.random.default_rng(13)

    ens = ensemble(T, L, N, seed=13, K_smooth=1, tau_seed=0.6)

    def avg_curve(ens_list):
        all_ds, all_ys = None, []
        for lat in ens_list:
            d, y = avg_abs_C2_vs_distance(lat, rmax=5)
            if all_ds is None:
                all_ds = d
            all_ys.append(y)
        mean_y = np.mean(np.vstack(all_ys), axis=0)
        return all_ds, mean_y

    ds0, y0 = avg_curve(ens)
    ens1 = [rg_step(lat, b=b, b_t=bt, tau=tau, rng=rng) for lat in ens]
    ds1, y1 = avg_curve(ens1)

    # BEFORE
    plt.figure()
    plt.semilogy(ds0, y0, marker="o")
    plt.xlabel("L1 distance d")
    plt.ylabel("average |C2(d)|")
    plt.title("Locality decay BEFORE RG (semi-log)")
    fig1 = os.path.join(OUT_DIR_FIGS, "locality_decay_before.png")
    plt.savefig(fig1, dpi=160)
    plt.close()

    # AFTER
    plt.figure()
    plt.semilogy(ds1, y1, marker="o")
    plt.xlabel("L1 distance d")
    plt.ylabel("average |C2(d)|")
    plt.title("Locality decay AFTER RG (semi-log)")
    fig2 = os.path.join(OUT_DIR_FIGS, "locality_decay_after.png")
    plt.savefig(fig2, dpi=160)
    plt.close()

    print("Saved:", fig1)
    print("Saved:", fig2)

if __name__ == "__main__":
    main()