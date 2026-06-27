"""
v2_firstpassage.py
==================
v2 — the first-passage discriminator.

v1 found the engine's STEADY-STATE occupancy is blind to channel length (Arm C: tau*
flat in L_ch) — a textbook consequence of detailed balance (occupancy ∝ exp(-βU_eff)
depends only on per-site free energies, not on inter-site path lengths). v2 tests the
complementary, KINETIC observable in a corridor where the corridor length is the genuine
diffusive bottleneck:

  Part A (THE discriminator):  thermal mean-first-passage-time MFPT(sep) ∝ sep²/D  while the
                               equilibrium occupancy P_R(sep) is FLAT. Kinetics sees the
                               corridor; occupancy cannot. (Plus FPT fluctuations / CV.)
  Part B (engine):             the engine occupancy (∝ exp(βT_c S_c)) is ALSO flat in sep
                               and favors the option-rich well — consistent with v1 in 1D.
  Part C (thermodynamic null): a 2D probability-current test with a positive control
                               (a rotational NESS shows a current; the conservative engine
                               force does not) — see v2_current.py.

The sharp thesis: engine and readout entropy are thermodynamically indistinguishable
(identical occupancy structure, zero current) yet kinetically distinguishable (first-passage
carries the corridor signature occupancy discards).

  python -m src.v2_firstpassage
"""
import os, json
import numpy as np

D, DT = 1.0, 0.01


# --------------------------------------------------------------------------- #
# Part A — mean first-passage time left-well -> midpoint, vs corridor length
# --------------------------------------------------------------------------- #
def mfpt_LtoR(lc, n_particles=5000, max_steps=200000, seed=0, x_target=0.0):
    """First-passage time from the left well (x=cL) to crossing x_target, under NATURAL
    overdamped Langevin (drift = -dU). Returns mean, std, CV, fraction passed."""
    rng = np.random.default_rng(seed)
    x = np.full(n_particles, lc.cL, dtype=float)
    sd = np.sqrt(2 * D * DT)
    fpt = np.full(n_particles, -1.0)
    active = np.ones(n_particles, dtype=bool)
    for t in range(1, max_steps + 1):
        idx = np.nonzero(active)[0]
        xa = x[idx]
        xa = xa - lc.dU(xa) * DT + sd * rng.standard_normal(xa.shape)
        xa = lc.reflect(xa)
        x[idx] = xa
        crossed = xa > x_target
        if crossed.any():
            hit = idx[crossed]
            fpt[hit] = t * DT
            active[hit] = False
            if not active.any():
                break
    passed = fpt > 0
    times = fpt[passed]
    return dict(mean=float(times.mean()), std=float(times.std()),
                cv=float(times.std() / times.mean()), frac_passed=float(passed.mean()),
                n=int(passed.sum()))


# --------------------------------------------------------------------------- #
# 1D causal path entropy field (engine), vectorized over grid points
# --------------------------------------------------------------------------- #
def sc_field_1d(lc, grid, n_sub, n_paths, seed, seeds=2, bins=44):
    rng_range = (lc.xl, lc.xr)
    sd = np.sqrt(2 * D * DT)
    out = np.zeros(len(grid))
    for s in range(seeds):
        rng = np.random.default_rng(seed + 131 * s)
        p = np.repeat(grid[:, None], n_paths, axis=1).astype(float)   # (G, n_paths)
        for _ in range(n_sub):
            p = p - lc.dU(p) * DT + sd * rng.standard_normal(p.shape)
            p = lc.reflect(p)
        for gi in range(len(grid)):
            h, _ = np.histogram(p[gi], bins=bins, range=rng_range)
            pr = h / h.sum(); pr = pr[pr > 0]
            out[gi] += float(-np.sum(pr * np.log(pr)))
    return out / seeds


def thermal_PR(lc, grid):
    """Equilibrium mass fraction in the right (option-rich) basin, x>0."""
    U = lc.U(grid); peq = np.exp(-(U - U.min()))
    Z = np.trapezoid(peq, grid)
    m = grid > 0.0
    return float(np.trapezoid(peq[m], grid[m]) / Z)


def engine_PR(lc, grid, n_sub, n_paths=1500, seed=0, contrast=3.0):
    sc = sc_field_1d(lc, grid, n_sub, n_paths, seed)
    beta_Tc = contrast / (sc.max() - sc.min() + 1e-12)
    peng = np.exp(beta_Tc * (sc - sc.max()))
    Z = np.trapezoid(peng, grid)
    m = grid > 0.0
    return float(np.trapezoid(peng[m], grid[m]) / Z), sc


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_v2(seps=(4.0, 5.0, 6.0, 7.0, 8.0), tau_engine=4.0, seed=0):
    from .landscape_v2 import Corridor1D
    print("\n=== v2 PART A/B: first-passage vs occupancy in a 1D corridor ===")
    print("    Prediction: MFPT ∝ sep²/D (kinetics sees corridor) while P_R is flat (occupancy blind).")
    n_sub = int(round(tau_engine / DT))
    rows = []
    for sep in seps:
        lc = Corridor1D(sep=float(sep))
        grid = np.linspace(lc.xl + 0.1, lc.xr - 0.1, 400)
        fp = mfpt_LtoR(lc, seed=seed)
        th = thermal_PR(lc, grid)
        en, _ = engine_PR(lc, grid, n_sub, seed=seed)
        rows.append(dict(sep=float(sep), sep2_over_D=sep * sep / D,
                         mfpt=fp["mean"], mfpt_std=fp["std"], cv=fp["cv"],
                         frac_passed=fp["frac_passed"], thermal_PR=th, engine_PR=en))
        print(f"  sep={sep:4.1f} sep²/D={sep*sep/D:5.1f} | MFPT={fp['mean']:7.2f} "
              f"CV={fp['cv']:.3f} passed={fp['frac_passed']:.2f} | P_R thermal={th:.3f} engine={en:.3f}")
    _fit_and_fig(rows)
    os.makedirs("results", exist_ok=True)
    np.savez("results/v2_firstpassage.npz", rows=json.dumps(rows), tau_engine=tau_engine)
    print("    data -> results/v2_firstpassage.npz")
    return rows


def _fit_and_fig(rows):
    x2 = np.array([r["sep2_over_D"] for r in rows])
    mfpt = np.array([r["mfpt"] for r in rows])
    th = np.array([r["thermal_PR"] for r in rows])
    en = np.array([r["engine_PR"] for r in rows])
    # MFPT = a + b*(sep²/D): affine fit (escape offset + diffusive corridor term)
    A = np.vstack([x2, np.ones_like(x2)]).T
    (b, a), *_ = np.linalg.lstsq(A, mfpt, rcond=None)
    pred = a + b * x2
    r2 = 1 - np.sum((mfpt - pred) ** 2) / (np.sum((mfpt - mfpt.mean()) ** 2) + 1e-12)
    print(f"  [Part A fit] MFPT = {b:.3f}*(sep²/D) + {a:.2f}   R²={r2:.3f}  (positive slope => kinetics sees L)")
    print(f"  [Part B] thermal P_R range [{th.min():.3f},{th.max():.3f}] (spread {th.max()-th.min():.3f}) "
          f"=> ~flat, occupancy blind to corridor length (detailed balance). "
          f"engine P_R range [{en.min():.3f},{en.max():.3f}] (spread {en.max()-en.min():.3f}) "
          f"=> weakly geometry-coupled (S_c sees the open plateau) but NOT the kinetic sep²/D signature.")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seps = np.array([r["sep"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax[0].plot(x2, mfpt, "o", color="#cc3a21", ms=9, label="MFPT (data)")
    xx = np.linspace(0, x2.max() * 1.05, 50)
    ax[0].plot(xx, a + b * xx, color="#1f6feb", label=f"fit: {b:.2f}·sep²/D + {a:.1f}  (R²={r2:.3f})")
    ax[0].set_xlabel("sep² / D   (diffusive corridor-traversal time)")
    ax[0].set_ylabel("mean first-passage time  L→mid")
    ax[0].set_title("v2 Part A: KINETICS sees the corridor\nMFPT ∝ sep²/D (the L-signature v1 occupancy missed)")
    ax[0].legend(fontsize=8.5)
    ax[1].plot(seps, th, "s-", color="#1f6feb", ms=8, label="thermal P_R — flat (detailed balance)")
    ax[1].plot(seps, en, "o-", color="#cc3a21", ms=8, label="engine P_R — weak S_c–geometry drift")
    ax[1].set_ylim(0, 1); ax[1].axhline(0.5, color="#aaa", ls=":")
    ax[1].set_xlabel("corridor length  sep")
    ax[1].set_ylabel("mass fraction in option-rich (right) well  P_R")
    ax[1].set_title("v2 Part B: OCCUPANCY carries no corridor-TIME signature\n"
                    "thermal flat (free energies only); engine only weakly geometry-coupled, not ∝sep²")
    ax[1].legend(fontsize=8.0)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/v2_firstpassage.png", dpi=150, facecolor="white")
    print("    figure -> figures/v2_firstpassage.png")


if __name__ == "__main__":
    run_v2()
