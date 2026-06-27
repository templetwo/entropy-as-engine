"""
v2_current.py
=============
v2 Part C — the broken-detailed-balance / probability-current test, with a positive control.

The prior-art review (docs/PRIOR_ART_AND_NOVELTY.md, §D) sharpens the deepest tension: the
position-only overdamped causal entropic force is CONSERVATIVE (Arm A: KL(driven‖analytic)
= 2e-4, curl-free), so its steady state is an equilibrium-like Boltzmann form with ZERO
probability current and ZERO entropy production. Under every standard non-equilibrium
detector (Battle–Broedersz probability currents, entropy-production rate, FDT violation) it
registers as "at equilibrium." So "engine" vs "readout" is thermodynamically
indistinguishable here — the distinction, if any, is purely kinetic (Part A).

This module demonstrates the detector on a positive control: a 2D harmonic well plus a
solenoidal force F_rot = ω(−y, x). ω=0 is conservative (equilibrium, zero current); ω>0 is a
genuine NESS with a circulating steady-state current. The scalar circulation we measure is
the mean angular current ⟨L_z⟩ = ⟨x·v_y − y·v_x⟩, which for this model equals ω·⟨r²⟩ — zero
iff detailed balance holds. The conservative engine force sits at the ω=0 end.

  python -m src.v2_current
"""
import os, json
import numpy as np

D, DT = 1.0, 0.01


def angular_current(omega, n=4000, steps=25000, burn=5000, k=1.0, box=3.0, seed=0, bin_every=25):
    """Mean angular current ⟨x·v_y − y·v_x⟩ for the rotor; ~0 iff detailed balance holds.
    Also returns a binned mean-velocity field for a current quiver (subsampled for speed)."""
    from .landscape_v2 import Rotor2D
    lc = Rotor2D(k=k, omega=omega, box=box)
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 0.5, size=(n, 2))
    sd = np.sqrt(2 * D * DT)
    Lz_sum, cnt = 0.0, 0
    nb = 24
    edges = np.linspace(-box, box, nb + 1)
    vx_acc = np.zeros((nb, nb)); vy_acc = np.zeros((nb, nb)); hit = np.zeros((nb, nb))
    for t in range(steps):
        drift = lc.drift(p)
        dp = drift * DT + sd * rng.standard_normal(p.shape)
        p_new = lc.reflect(p + dp)
        v = (p_new - p) / DT
        if t >= burn:
            Lz_sum += float(np.sum(p[:, 0] * v[:, 1] - p[:, 1] * v[:, 0])); cnt += n
            if t % bin_every == 0:   # quiver field only (subsampled) — the scatter is the slow part
                ix = np.clip(np.digitize(p[:, 0], edges) - 1, 0, nb - 1)
                iy = np.clip(np.digitize(p[:, 1], edges) - 1, 0, nb - 1)
                np.add.at(vx_acc, (ix, iy), v[:, 0]); np.add.at(vy_acc, (ix, iy), v[:, 1])
                np.add.at(hit, (ix, iy), 1.0)
        p = p_new
    Lz = Lz_sum / max(cnt, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mvx = np.where(hit > 0, vx_acc / hit, 0.0)
        mvy = np.where(hit > 0, vy_acc / hit, 0.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return dict(omega=float(omega), Lz=Lz, centers=centers, mvx=mvx, mvy=mvy, hit=hit)


def run_current(omegas=(0.0, 0.5, 1.0, 2.0), seed=0):
    print("\n=== v2 PART C: probability-current / broken-detailed-balance test (positive control) ===")
    print("    ⟨L_z⟩ ≈ 0  => detailed balance / zero current (the conservative engine's regime).")
    res = [angular_current(w, seed=seed) for w in omegas]
    rows = [dict(omega=r["omega"], Lz=r["Lz"]) for r in res]
    for r in rows:
        tag = "  <- equilibrium / zero current (engine sits here)" if abs(r["omega"]) < 1e-9 else ""
        print(f"  omega={r['omega']:4.1f}  <L_z>={r['Lz']:+.4f}{tag}")
    print("    Interpretation: the causal entropic force is conservative (Arm A) => omega=0 column")
    print("    => zero current/entropy production => engine is thermodynamically indistinguishable")
    print("    from a readout relaxation. The engine/readout distinction is therefore KINETIC (Part A).")
    _fig(res, rows)
    os.makedirs("results", exist_ok=True)
    np.savez("results/v2_current.npz", rows=json.dumps(rows))
    print("    data -> results/v2_current.npz")
    return rows


def _fig(res, rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8))
    om = np.array([r["omega"] for r in rows]); lz = np.array([r["Lz"] for r in rows])
    ax[0].plot(om, lz, "o-", color="#cc3a21", ms=9)
    ax[0].axhline(0, color="#888", ls=":")
    ax[0].set_xlabel("rotational drive  ω  (ω=0 = conservative)")
    ax[0].set_ylabel("mean angular current  ⟨L_z⟩")
    ax[0].set_title("Current detector vs drive\n⟨L_z⟩=0 at ω=0 (engine's regime); rises with drive")
    for col, r in [(1, res[0]), (2, res[-1])]:
        X, Y = np.meshgrid(r["centers"], r["centers"], indexing="ij")
        m = r["hit"] > r["hit"].max() * 0.02
        ax[col].quiver(X[m], Y[m], r["mvx"][m], r["mvy"][m], color="#1f6feb", scale=30)
        ax[col].set_title(f"steady-state current field, ω={r['omega']:.1f}\n"
                          + ("no circulation (equilibrium)" if r["omega"] == 0 else "circulating current (NESS)"))
        ax[col].set_xlabel("x"); ax[col].set_ylabel("y"); ax[col].set_aspect("equal")
    fig.suptitle("v2 Part C: the engine force is conservative (Arm A) → it sits at ω=0 → zero current "
                 "→ thermodynamically indistinguishable from equilibrium; the discriminator must be kinetic",
                 fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/v2_current.png", dpi=140, facecolor="white")
    print("    figure -> figures/v2_current.png")


if __name__ == "__main__":
    run_current()
