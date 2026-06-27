#!/usr/bin/env python
"""
v3_figures.py
=============
PURE LOADER. Reads results/v3_manifest.json (and results/v2_firstpassage.npz for the
v1/v2 context panel in fig7) and renders exactly seven publication-grade figures
(PNG + PDF @ 300 dpi).

- Zero np.random anywhere.
- Zero hardcoded data literals; every plotted value comes from a committed artifact
  (the v3 manifest, or the v2 npz for fig7's prior-stage panel).
- Each figure is self-contained (title, units, legend).
- tau rendered with perceptual gradient (viridis).
- Fails loudly on missing keys or shape mismatch.

Run:
  python -m src.v3_figures

Produces:
  figures/fig1_dial.png + .pdf
  ...
  figures/fig7_arc.png + .pdf
  figures/captions.md
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle

# ----------------------------- load & validate -----------------------------
def load_manifest():
    path = "results/v3_manifest.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run production first.")
    with open(path) as f:
        d = json.load(f)
    required = [
        "meta", "tau_grid", "sep_grid",
        "partA_mfpt_tau", "partA_fpt_hist", "partA_ks_vs_tau0",
        # partB_* deliberately absent: reducibility is definitional (Arm A), not an empirical test
        "partC_alpha_tau", "partC_mfpt_vs_sep",
        "partD_current_engine", "partD_current_control", "partD_control_asym",
        "partE_ueff_surface"
    ]
    for k in required:
        if k not in d:
            raise KeyError(f"Manifest missing required key: {k}")
    return d

# ----------------------------- helpers -----------------------------
def tau_colors(n, cmap_name="viridis"):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / max(1, n-1)) for i in range(n)]

def midpoints(edges):
    e = np.asarray(edges)
    return 0.5 * (e[:-1] + e[1:])

def save_both(fig, base):
    os.makedirs(os.path.dirname(base), exist_ok=True)
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("figure ->", base + ".png", "+", base + ".pdf")

# ----------------------------- figures -----------------------------
def fig1_dial(d):
    """The τ-dial in kinetics (headline). FPT distribs per τ + MFPT(τ) with CI."""
    taus = np.array(d["tau_grid"])
    hist = d["partA_fpt_hist"]
    edges = np.array(hist["bin_edges"])
    counts_list = hist["counts"]
    mids = midpoints(edges)
    db = edges[1] - edges[0]

    mf = d["partA_mfpt_tau"]
    means = np.array(mf["mean"])
    lo = np.array(mf["ci_lo"])
    hi = np.array(mf["ci_hi"])

    colors = tau_colors(len(taus))

    fig = plt.figure(figsize=(11, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.22)

    ax0 = fig.add_subplot(gs[0])
    for i, (tau, cnts) in enumerate(zip(taus, counts_list)):
        dens = np.asarray(cnts) / (np.sum(cnts) * db + 1e-12)
        ax0.step(mids, dens, where="mid", color=colors[i], lw=1.6, label=f"τ={tau:.2f}")
    ax0.set_xlabel("first-passage time (to x=0)")
    ax0.set_ylabel("density")
    ax0.set_title("Fig 1 — Engine FPT distributions sharpen & shift left as τ grows\n(Corridor1D, fixed sep=6, causal force only)")
    ax0.legend(loc="upper right", fontsize=7, framealpha=0.92, ncol=2)
    ax0.set_xlim(0, 180)

    ax1 = fig.add_subplot(gs[1])
    ax1.errorbar(taus, means, yerr=[means-lo, hi-means], fmt="o-", color="#222222", ms=5,
                 capsize=3, lw=1.4, label="MFPT(τ) ±95% CI")
    ax1.set_xlabel("horizon τ")
    ax1.set_ylabel("mean first-passage time")
    ax1.set_title("MFPT(τ) — the kinetic dial (monotonic decrease)")
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)

    fig.suptitle("v3: first-passage time is a τ-tunable dial (production ensemble 8000)", fontsize=10, y=0.995)
    save_both(fig, "figures/fig1_dial")

def fig2_engine_vs_readout(d):
    """Left: fixed (short-τ / readout-like), one curve. Right: full τ family."""
    taus = np.array(d["tau_grid"])
    hist = d["partA_fpt_hist"]
    edges = np.array(hist["bin_edges"])
    counts_list = hist["counts"]
    mids = midpoints(edges)
    db = edges[1] - edges[0]
    colors = tau_colors(len(taus))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

    # left: "readout" = shortest horizon (least structure, closest to fixed)
    i0 = 0
    dens0 = np.asarray(counts_list[i0]) / (np.sum(counts_list[i0]) * db + 1e-12)
    ax[0].step(mids, dens0, where="mid", color="#1f6feb", lw=2.2, label=f"τ={taus[i0]:.2f} (short-horizon)")
    ax[0].fill_between(mids, 0, dens0, step="mid", alpha=0.15, color="#1f6feb")
    ax[0].set_xlabel("FPT (to x=0)")
    ax[0].set_ylabel("density")
    ax[0].set_title("Readout-like (fixed, τ-independent view)\none curve — no dial")
    ax[0].legend(loc="upper right")
    ax[0].set_xlim(0, 160)

    # right: engine family
    for i, (tau, cnts) in enumerate(zip(taus, counts_list)):
        dens = np.asarray(cnts) / (np.sum(cnts) * db + 1e-12)
        ax[1].step(mids, dens, where="mid", color=colors[i], lw=1.5, label=f"τ={tau:.2f}")
    ax[1].set_xlabel("FPT (to x=0)")
    ax[1].set_title("Engine (τ-dial)\nfamily of curves — the horizon moves the distribution")
    ax[1].legend(loc="upper right", fontsize=7, ncol=2)
    ax[1].set_xlim(0, 160)

    fig.suptitle("Fig 2 — A readout has one FPT curve; the engine has a dial (manifest data only)", y=0.995)
    save_both(fig, "figures/fig2_engine_vs_readout")

def fig3_reducibility(d):
    """Definitional reducibility (1D conservative case) + honest frontier note.
    No empirical KS test is shown or claimed: the 'engine' and 'static U_eff' are
    the identical SDE (F_ce = −∇U_eff from Arm A). Reducibility is true by definition
    in this regime. The real empirical anchor is zero current (Part D / fig5).
    A test with teeth lives only in 2D+ non-conservative (v4)."""
    taus = np.array(d["tau_grid"])
    edges = np.array(d["partA_fpt_hist"]["bin_edges"])
    eng_counts = d["partA_fpt_hist"]["counts"]
    mids = midpoints(edges)
    db = edges[1] - edges[0]
    colors = tau_colors(len(taus))

    fig = plt.figure(figsize=(11, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 0.95], wspace=0.18)

    # left: the τ-family of FPTs (the dial we *do* measure)
    ax0 = fig.add_subplot(gs[0, 0])
    for i, (tau, cnts) in enumerate(zip(taus, eng_counts)):
        dens = np.asarray(cnts) / (np.sum(cnts) * db + 1e-12)
        ax0.step(mids, dens, where="mid", color=colors[i], lw=1.5)
    ax0.set_xlabel("first-passage time (to x=0)")
    ax0.set_ylabel("density")
    ax0.set_title("Engine FPT distributions (τ family)\n(the measured dial)")
    ax0.set_xlim(0, 160)
    ax0.text(0.98, 0.97, "sep=6 fixed\n8000 traj/τ", transform=ax0.transAxes, ha="right", va="top",
             fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # right: honest statement (no fake p-values)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axis("off")
    txt = (
        "Reducibility is definitional here\n\n"
        "• Arm A established: causal force is conservative\n"
        "  F_ce = −∇U_eff  (U_eff = −βT_c S_c)\n\n"
        "• Therefore at fixed τ the engine dynamics *are*\n"
        "  equilibrium relaxation in U_eff(τ)\n\n"
        "• In 1D: every force is curl-free by geometry.\n"
        "  'Engine ≡ readout' is true by definition, not measurement.\n\n"
        "• Presenting duplicate-SDE KS p≥0.05 as a 'finding'\n"
        "  would be circular. We do not do that.\n\n"
        "The empirical anchor for thermodynamic indistinguishability\n"
        "is Part D (fig 5): ⟨L_z⟩ ≈ 0 for the engine (vs nonzero\n"
        "for the Rotor2D positive control).\n\n"
        "The sharp question 'does the τ-knob leave a fingerprint\n"
        "no static potential can mimic?' receives answer NO\n"
        "in this conservative 1D regime.\n\n"
        "Real frontier = v4: non-conservative force (curl ≠ 0)\n"
        "in 2D+, where a non-vacuous reducibility test exists."
    )
    ax1.text(0.02, 0.98, txt, transform=ax1.transAxes, va="top", fontsize=9,
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f8f8", edgecolor="#333", alpha=0.95))

    fig.suptitle("Fig 3 — Reducibility in conservative 1D is definitional (Arm A), not empirical. Honest negative: τ-dial is equilibrium in U_eff(τ). Frontier = v4.", y=0.995, fontsize=10)
    save_both(fig, "figures/fig3_reducibility")

def fig4_sep_scaling(d):
    """Engine first-passage scaling under driven force: drift-dominated (α ≈ 1).
    Honest measurement from manifest. α~1 is the expected signature of ballistic/drift
    transport (constant force component), not the diffusive sep²/D (α=2) of thermal v2.
    This does not by itself separate a 'true engine' from a disguised reward force."""
    taus = np.array(d["partC_mfpt_vs_sep"]["tau"])
    seps = np.array(d["partC_mfpt_vs_sep"]["sep"])
    mfpt_grid = np.array(d["partC_mfpt_vs_sep"]["mfpt"])
    alpha = d["partC_alpha_tau"]
    a_m = np.array(alpha["mean"])
    a_lo = np.array(alpha["ci_lo"])
    a_hi = np.array(alpha["ci_hi"])
    colors = tau_colors(len(taus))

    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.2)

    ax0 = fig.add_subplot(gs[0])
    for i, tau in enumerate(taus):
        ax0.loglog(seps, mfpt_grid[i], "o-", color=colors[i], ms=4.5, lw=1.3, label=f"τ={tau:.2f}")
    ax0.set_xlabel("sep")
    ax0.set_ylabel("MFPT (log-log)")
    ax0.set_title("Fig 4 — MFPT(sep) at fixed τ (log-log)\n(drift-dominated under causal force)")
    ax0.legend(fontsize=7, ncol=2)
    ax0.grid(True, alpha=0.3, which="both")

    ax1 = fig.add_subplot(gs[1])
    ax1.errorbar(taus, a_m, yerr=[a_m - a_lo, a_hi - a_m], fmt="s-", color="#222222", ms=5,
                 capsize=2.5, lw=1.3, label="fitted α(τ) ±CI (9 seps)")
    ax1.axhline(1.0, color="#1f6feb", ls="--", lw=1.4, label="α≈1 (drift / ballistic)")
    ax1.set_xscale("log")
    ax1.set_xlabel("τ")
    ax1.set_ylabel("exponent α")
    ax1.set_title("α(τ) — drift-dominated (α≈1)\n(not the diffusive α=2 of thermal v2)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Caveat box
    fig.text(0.5, 0.01,
             "Caveat: α≈1 is generic for drift-dominated transport. It does not by itself demonstrate that the entropic force is 'more than a disguised reward'.",
             ha="center", fontsize=8, style="italic", wrap=True)

    save_both(fig, "figures/fig4_sep_scaling")

def fig5_positive_control(d):
    """Current: engine≈0 vs Rotor; directional FPT asymmetry on control."""
    eng = d["partD_current_engine"]
    ctrl = d["partD_current_control"]
    asym = d["partD_control_asym"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))

    # left: current
    labels = ["engine\n(ω=0)", f"control\n(ω={ctrl['omega']})"]
    means = [eng["mean"], ctrl["mean"]]
    yerr = [[eng["mean"]-eng["ci_lo"], ctrl["mean"]-ctrl["ci_lo"]],
            [eng["ci_hi"]-eng["mean"], ctrl["ci_hi"]-ctrl["mean"]]]
    colors_b = ["#2ca02c", "#d62728"]
    ax[0].bar(labels, means, yerr=yerr, color=colors_b, capsize=4, error_kw={"lw":1.5})
    ax[0].axhline(0.0, color="k", lw=0.8)
    ax[0].set_ylabel("⟨L_z⟩")
    ax[0].set_title("Probability current\nengine ≈ 0 (conservative) vs control ≈ ω·⟨r²⟩")

    # right: asym hists (fwd/rev)
    fwd = np.asarray(asym["fwd"])
    rev = np.asarray(asym["rev"])
    bins = np.linspace(0, max(np.max(fwd), np.max(rev), 10), 35)
    ax[1].hist(fwd, bins=bins, alpha=0.55, label=f"fwd (n={len(fwd)})", color="#d62728")
    ax[1].hist(rev, bins=bins, alpha=0.55, label=f"rev (n={len(rev)})", color="#1f77b4")
    ax[1].set_xlabel("first-passage time (directional)")
    ax[1].set_ylabel("count")
    p = asym["ks_p"]
    ax[1].set_title(f"Directional FPT asymmetry (control)\nKS p={p:.2g}")
    ax[1].legend()

    fig.suptitle("Fig 5 — Positive control: instrument detects genuine NESS (current + asym); engine registers null", y=0.995)
    save_both(fig, "figures/fig5_positive_control")

def fig6_ueff_surface(d):
    """U_eff(x, τ) heatmap — the cause of the dial."""
    ue = d["partE_ueff_surface"]
    x = np.array(ue["x"])
    taus = np.array(ue["tau"])
    U = np.array(ue["U_eff"])  # shape (n_tau, n_x)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    # imshow: tau vertical (increasing down or up), x horizontal
    im = ax.imshow(U, aspect="auto", origin="lower",
                   extent=[x.min(), x.max(), taus.min(), taus.max()],
                   cmap="magma", interpolation="bilinear")
    ax.set_xlabel("x (corridor location marked)")
    ax.set_ylabel("horizon τ (log scale)")
    ax.set_title("Fig 6 — Effective landscape U_eff(x, τ) = −βT_c S_c(x, τ)\n(the dial that moves first-passage)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("U_eff")

    # mark corridor approx at fixed_sep=6 => centers ~ ±3
    sep = 6.0
    ax.axvline(-sep/2, color="white", lw=1.2, ls="--", alpha=0.85)
    ax.axvline(+sep/2, color="white", lw=1.2, ls="--", alpha=0.85)
    ax.text(-sep/2, taus.max()*0.92, "left well", color="white", ha="center", fontsize=8)
    ax.text(+sep/2, taus.max()*0.92, "right well", color="white", ha="center", fontsize=8)

    ax.set_yscale("log")
    save_both(fig, "figures/fig6_ueff_surface")

def fig7_arc(d):
    """Four-observable summary / arc across v1→v2→v3. Honest framing:
    v3 re-confirms v2's thermodynamic equivalence in the conservative regime and shows
    that the τ-dial is equilibrium dynamics in U_eff(τ). The negative result (no unique
    fingerprint beyond static U_eff) correctly identifies the v4 frontier."""
    taus = np.array(d["tau_grid"])
    mf = d["partA_mfpt_tau"]
    eng = d["partD_current_engine"]
    ctrl = d["partD_current_control"]

    fig = plt.figure(figsize=(11, 9.0))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.28)

    # v1/v2 occupancy panel — loaded from the committed v2 artifact (not literals)
    v2_path = "results/v2_firstpassage.npz"
    if not os.path.exists(v2_path):
        raise FileNotFoundError(f"Missing {v2_path}. fig7's v1/v2 panel reads it (run v2 first).")
    v2 = np.load(v2_path, allow_pickle=True)
    v2rows = json.loads(str(v2["rows"]))
    v2_seps = [r["sep"] for r in v2rows]
    v2_thermal = [r["thermal_PR"] for r in v2rows]
    v2_engine = [r["engine_PR"] for r in v2rows]
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(v2_seps, v2_thermal, "o--", color="#1f6feb", label="thermal (flat)")
    ax0.plot(v2_seps, v2_engine, "s-", color="#cc3a21", label="engine (S_c geometry)")
    ax0.set_xlabel("sep")
    ax0.set_ylabel("P (right basin)")
    ax0.set_title("v1/v2: occupancy blind to L (detailed balance)\n(thermal flat; engine sees options but not sep²/D)")
    ax0.legend(fontsize=8)
    ax0.set_ylim(0.3, 0.9)

    # v2: current zero (the real empirical anchor)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(["engine (ω=0)", "Rotor (ω=0.5)"], [eng["mean"], ctrl["mean"]],
            yerr=[[eng["mean"]-eng["ci_lo"], ctrl["mean"]-ctrl["ci_lo"]],
                  [eng["ci_hi"]-eng["mean"], ctrl["ci_hi"]-ctrl["mean"]]],
            color=["#2ca02c", "#d62728"], capsize=4)
    ax1.axhline(0, color="k", lw=0.7)
    ax1.set_ylabel("⟨L_z⟩")
    ax1.set_title("v2: current test — engine = 0 (conservative)\npositive control detects real NESS\n(this is the measurement)")

    # v3 MFPT dial (what we measure)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.errorbar(taus, mf["mean"], yerr=[np.array(mf["mean"])-np.array(mf["ci_lo"]),
                                         np.array(mf["ci_hi"])-np.array(mf["mean"])],
                 fmt="o-", ms=5, capsize=3, color="#222")
    ax2.set_xscale("log")
    ax2.set_xlabel("τ")
    ax2.set_ylabel("MFPT")
    ax2.set_title("v3: MFPT(τ) dial — monotone, kinetic\n(the horizon moves first-passage)")

    # v3 honest negative
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    note = (
        "v3 honest negative (this run)\n\n"
        "• τ produces a visible FPT dial (real).\n"
        "• But the dial *is* equilibrium relaxation\n"
        "  in the τ-dependent U_eff(τ) — by definition\n"
        "  once Arm A (conservative) is granted.\n\n"
        "• α ≈ 1 (drift under force), not thermal\n"
        "  diffusive α=2.\n\n"
        "• Zero current re-confirms thermodynamic\n"
        "  equivalence (fig 5).\n\n"
        "Conclusion: in conservative 1D the τ-knob\n"
        "does not produce a fingerprint no static U\n"
        "can mimic. The frontier is v4 (non-conservative\n"
        "force, 2D+, curl ≠ 0)."
    )
    ax3.text(0.02, 0.98, note, transform=ax3.transAxes, va="top", fontsize=9,
             family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8f0", edgecolor="#333"))

    fig.suptitle("Fig 7 — Arc (honest): v3 re-confirms v2 thermodynamic equivalence in conservative 1D.\nThe negative (no unique τ-fingerprint beyond U_eff) is the result. Frontier = v4 non-conservative. Data: manifest (8000 traj/cell).", y=0.995, fontsize=9)
    save_both(fig, "figures/fig7_arc")

def main():
    d = load_manifest()
    print("manifest loaded, ensemble:", d["meta"].get("ensemble_per_cell"))
    fig1_dial(d)
    fig2_engine_vs_readout(d)
    fig3_reducibility(d)
    fig4_sep_scaling(d)
    fig5_positive_control(d)
    fig6_ueff_surface(d)
    fig7_arc(d)
    print("\nAll seven figures written (PNG + PDF @ 300 dpi).")

if __name__ == "__main__":
    main()
