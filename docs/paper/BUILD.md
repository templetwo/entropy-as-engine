# Paper build — `entropy-as-engine`

This directory holds the manuscript for the v3 result:
**"Entropy as a Tunable Equilibrium, Not an Engine."**

## Where it goes

Place at `docs/paper/paper.tex` in the repo. The LaTeX uses

```latex
\graphicspath{{../figures/}{figures/}{../../figures/}}
```

so it finds the committed figure PNGs in `figures/` whether you compile from
the repo root or from `docs/paper/`. **No figures are bundled in this file** —
it references the real committed images by filename, so the genuine figures
appear on compile. Nothing here is regenerated or synthetic.

## Build

```bash
cd docs/paper
pdflatex paper.tex      # run twice for cross-refs + hyperlinks
pdflatex paper.tex
# or:
latexmk -pdf paper.tex
```

Output: `paper.pdf` with the real figures embedded.

If you don't have a local TeX distribution: drop `paper.tex` + the `figures/`
PNGs into Overleaf and compile there.

## Figures the manuscript references (reconcile against the merged repo)

The paper embeds six figures. Confirm these filenames match what actually
landed on `main` after the v3-confirmation merge, and rename in the `.tex` if
the committed names differ:

| In paper | Expected file | Carries |
|---|---|---|
| Fig 1 | `fig1_dial.png` | FPT distributions shift with τ + MFPT(τ) — the kinetic dial |
| Fig 2 | `fig2_engine_vs_readout.png` | one readout curve vs the engine τ-family |
| Fig 3 (control) | `fig5_positive_control.png` | currents (engine≈0 vs rotor) + directional FPT asymmetry |
| Fig 4 (scaling) | `fig4_sep_scaling.png` | α≈1 drift, **not** α≈2 diffusion |
| Fig 5 (surface) | `fig6_ueff_surface.png` | U_eff(x,τ) landscape reshaping with the dial |
| Fig 6 (arc) | `fig7_arc.png` | check/cross summary across observables |

Note: the original `fig3_reducibility.png` is **deliberately not embedded** —
the reducibility point is now made as a definitional argument in the text
(§5.5), matching the review that removed the circular Part B test. If a
reframed `fig3` survived on `main`, decide whether to add it back as an
illustration of the definitional point; it is not required by the argument.

## Two things to verify before submission (verify-before-declare)

1. **Figure filenames** — as above, against the actual merged `main`.
2. **Bibliography** — the reference list was written from memory. Confirm
   volumes/pages/years against the primary sources (the paper says so
   explicitly in a footnote). The two I'm most confident on are
   Wissner-Gross & Freer, *PRL* **110**, 168702 (2013) and
   Still et al., *PRL* **109**, 120604 (2012); verify the rest.

## Numbers in the paper (all from the verified runs, for cross-check)

- Phase 0: engine mass 0.483 → 0.668 as τ grows; control fixed 0.527; KL ≈ 0.8–1.1
- Arm A: KL(driven‖analytic) = 0.0002, overlap 0.993
- Arm B: engine P_L 0.51 → 0.91; control 0.371 (flat)
- Arm C: τ* flat in L_ch; D-scan τ* ∝ 1/D (R²=0.97) — steady-state discriminator falsified
- v2 thermal MFPT ∝ sep²/D, R²=0.992 (this is the *thermal* α=2, not the engine)
- Engine current ⟨L_z⟩ ≈ 1×10⁻³ (≈0); Rotor ω=0.5 control ⟨L_z⟩ ≈ 1 (≈2ω)
- Control directional FPT asymmetry p ≈ 2.5×10⁻⁵
- Engine scaling α ≈ 0.66–1.08 (≈0.85), drift-dominated; fit poorly conditioned (5 lengths)
- Production sweep: 8×10³ traj/cell, 8×5 grid, seed 42
- Units: T=1, γ=1, D=1, dt=0.01
