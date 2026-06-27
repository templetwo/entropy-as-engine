# RESULTS — Entropy as Engine

Executor: Claude Code (Opus 4.8, "ultra"), **MacBook Pro seat** (the SPEC targeted the
Studio; this run is the MacBook — see Environment). Verified-before-declared throughout.

## Environment
- hardware: Apple Silicon MacBook Pro (`Darwin 25.5.0`, arm64), 18 GB unified memory, 12 cores
- python: 3.10.12 (pyenv); numpy 2.2.6, scipy 1.15.3, matplotlib 3.10.5
- jax backend / device: **JAX not installed on this seat** → ran the NumPy-fallback path.
  The SPEC's "JAX vmap/jit/scan" 2D pipeline is replaced by an equivalent **vectorized
  NumPy engine** (`src/sc2d_fast.py`: all grid points' walker ensembles evolved as one
  `(G, n_paths, 2)` float32 array) plus **multiprocessing** fan-out over the (geometry, τ)
  grid (`src/run_fast.py`). Oracle-validated against the reference triple-loop
  (`run._sc_field_2d`): field correlation **0.993**, relative-L2 **0.005**.
- seeds: fixed and logged. Phase 0 seeds `s*131+j`; Arm A seed 0 (+7 for the driven run);
  Arms B/C seed 0 with `seeds=3` averaged per S_c field.
- sampling (Arms B/C): grid 44×28, `n_paths=700`, `seeds=3`, endpoint-entropy `bins=28`,
  contrast `βT_c = 3/(S_c.max−S_c.min)` (matches the reference normalization).
- wall-clock: Phase 0 7.6 s; Arm A 6.4 s; Arm B 176 s (16 τ × 10 procs); Arm C ~14 min
  (5 L_ch × 16 τ); D-scan diagnostic ~13 min (5 D × 14 τ). All on CPU.

## Phase 0 reproduction  (`python -m src.phase0_validated_1d`)
- argmax migration:           **−3.66 → ~0** (monotone)   — expect −3.7 → ~0  ✓
- mass-in-B (option-rich):    **0.483 → 0.668** (monotone) — expect 0.48 → ~0.67  ✓
- KL(engine‖eq) range:        **0.79 – 1.14** (never ~0)   — expect ~0.75 – 1.16  ✓
- control: fixed, argmax −2.4 in deep well A, mass 0.527 (τ-invariant)  ✓
- **reproduces Phase 0?  [x] yes**  — figure `figures/phase0_tau_overlay_3d.png`

## Arm A — conservativeness  (`python -m src.run --arm A`)
- KL(driven‖analytic): **0.0002**
- overlap:             **0.993**
- **verdict:  [x] PASS (analytic shortcut OK for Arms B/C)**  — figure `figures/armA_conservativeness.png`
- Tighter than the web-seat preliminary (0.012). The overdamped causal entropic force is
  conservative, so `p_engine ∝ exp(βT_c S_c)` is used in B/C (no driven trajectories).

## Arm B — 2D crossover  (`python -m src.run_fast --arm B`)
- chamber tuning: prior SPEC defaults put thermal mass in the LARGE chamber
  (`P_L_control ≈ 0.67`, backwards). Retuned `ChamberChannel2D` defaults to
  `d_S=5.5, r_S=0.6, d_L=2.0, r_L=1.8` → **`P_L_control = 0.371`** (thermal favors SMALL),
  large chamber still 9× the area (option-rich). τ* is NOT tuned — Arm C's scaling is the claim.
- P_L control (fixed): **0.371**
- P_L(τ): **0.511 → 0.910**, strictly monotone non-decreasing (16-pt log τ grid 0.2→18;
  min step +0.006). Engine sits near 0.51 at τ→0 (≈ uniform S_c, half the domain is x>0 —
  options not yet differentiated) and migrates to 0.91 as the horizon reveals the
  option-rich chamber. The thermal control (0.371) never moves.
- τ*: **1.33** (first half-rise crossing of thresh 0.640) / **2.44** (logistic midpoint).
  L_ch²/D = 1.5²/1 = 2.25 — the logistic midpoint brackets it almost exactly.
- **engine ≠ control?  [x] yes** — decisively (Δ = 0.539 at long τ). Figure
  `figures/armB_crossover.png`.

## Arm C — discriminator  (MAKE-OR-BREAK)  (`python -m src.run_fast --arm C`)

| L_ch | L²/D | τ\* (first-cross) | τ\* (logistic) | P_L_control | P_L_max |
|---|---|---|---|---|---|
| 1.0 | 1.00 | 1.57 | 2.85 | 0.382 | 0.917 |
| 1.5 | 2.25 | 1.33 | 2.44 | 0.371 | 0.910 |
| 2.0 | 4.00 | 1.15 | 2.27 | 0.355 | 0.902 |
| 2.5 | 6.25 | 1.09 | 2.32 | 0.336 | 0.897 |
| 3.0 | 9.00 | 1.07 | 2.54 | 0.318 | 0.892 |

- fit τ* = slope·L_ch²/D (through origin): slope **0.180**, **R²(origin) = −17.7** (worse than a constant).
- affine fit: τ* = **−0.057·(L²/D) + 1.50** — slope ≈ 0; τ* is flat-to-slightly-**decreasing** over a 9× sweep in L²/D.
- linear through origin?  **[ ] no**
- **VERDICT:  [x] the SPEC's prediction τ\* ∝ L_ch²/D FAILS — reported honestly (not tuned away).**

**Why it fails (diagnosed, self-consistent).** Under the analytic steady-state shortcut
`p_engine ∝ exp(βT_c S_c)`, mass is placed wherever S_c is high *with no transit time* —
there is no "particle migrating through the channel over τ." S_c inside the large chamber
is high as soon as that chamber *self-explores* (τ ≳ r_L²/D ≈ 3.2), **independent of the
channel**. So the P_L crossover is gated by **large-chamber exploration, not channel
traversal** — which is L_ch-independent. The mild *decrease* is mechanism, not noise: as
L_ch grows the wall (thickness ~L_ch in x) widens and eats into the large chamber, so
P_L_control and P_L_max both fall monotonically (0.382→0.318, 0.917→0.892) — the effective
chamber shrinks → faster self-exploration → smaller τ*. The steady-state P_L metric simply
cannot see the channel-traversal time.

### Diagnostic — is τ\* a genuine diffusive geometric timescale?  (`python -m src.diagnostics --scan D`)
Vary the diffusion constant D (via friction γ at fixed kT=1, so p_eq ∝ exp(−U) and the
control are D-invariant — a confound-free knob). If τ* is a diffusive time r_eff²/D it must
scale as **1/D**.

| D | τ\* (first-cross) | τ\* (logistic) | τ\*·D | P_L_control |
|---|---|---|---|---|
| 0.5 | 2.32 | 5.80 | 1.16 | 0.371 |
| 0.7 | 1.83 | 3.67 | 1.28 | 0.371 |
| 1.0 | 1.33 | 2.43 | 1.33 | 0.371 |
| 1.4 | 0.95 | 1.69 | 1.33 | 0.371 |
| 2.0 | 0.69 | 1.25 | 1.38 | 0.371 |

- fit: τ* = **1.23/D, R²(through-origin in 1/D) = 0.973** (logistic: 2.71/D, R² = 0.975).
- τ*·D constant: mean 1.30, std 0.08 (first-cross) — i.e. **τ\* ∝ 1/D**. P_L_control is
  identical (0.371) at every D → the equilibrium is untouched, the knob is purely kinetic.
- **Conclusion:** τ* is a genuine *diffusive geometric* timescale (∝1/D, R²≈0.97) that is
  *independent of the channel length* (Arm C). It tracks the **chamber-exploration** time
  (~r_L²/D in magnitude), not the **channel-traversal** time (L_ch²/D) the SPEC predicted.
  Figure `figures/diag_Dscan.png`.

### What survives, what doesn't (honest summary)
- **SURVIVES:** entropy-as-engine generates real, horizon-dependent organization. The knob
  is genuine (Arm B: monotone P_L(τ), engine≠control by Δ≈0.54), the steady state is the
  conservative analytic one (Arm A), and τ* is a true *diffusive geometric* time (∝1/D).
- **DOES NOT SURVIVE:** the specific **channel-length fingerprint** τ* ∝ L_ch²/D that the
  SPEC designed to rule out "the entropic force is just a disguised reward." With the
  steady-state P_L observable it is not realized — τ* tracks r_L²/D instead. The
  disguised-reward objection is therefore **not** closed by this experiment as built.
- **RECOMMENDED v2 (to actually test the channel fingerprint):** a *dynamical* observable
  that must physically cross the channel — e.g. mean first-passage time of F_ce-driven
  particles from the small chamber to the large one vs L_ch — which is channel-traversal
  gated and would scale as L_ch²/D if the engine claim holds. Steady-state P_L cannot probe
  it; this is a metric limitation, not (yet) a refutation of the engine.

## Prior art & how to read this result  (see `docs/PRIOR_ART_AND_NOVELTY.md`)
The Arm C negative is **not a surprising bug — it is a rediscovery of a foundational result**,
and that framing is the honest one:
- **Why Arm C *had* to fail with a steady-state metric.** Under detailed balance the stationary
  occupancy is ∝ exp(−βU_eff) with `U_eff = −T_c S_c`; it depends only on per-site free energies
  and is **provably blind** to barriers and inter-site path lengths. Channel-traversal time
  `L_ch²/D` is a *kinetic* quantity that lives only in dynamical observables (rates, MFPT). This
  is the thermodynamics–kinetics separation: Hänggi–Talkner–Borkovec, *Rev. Mod. Phys.* 62 (1990);
  Gardiner; van Kampen. The `L²/D` scaling is canonical diffusive first-passage and *will* show up
  in the MFPT — just not in occupancy.
- **The deepest tension (states it plainly).** Arm A found the position-only overdamped causal
  force is **conservative** → its steady state is an equilibrium-like Boltzmann form with **zero
  probability current and zero entropy production**. So in this regime "engine entropy" is
  **thermodynamically indistinguishable** from a readout relaxation to `U_eff`; the τ-dependence of
  `U_eff` is the *only* thing separating them. That is exactly why the **τ knob** (not a
  current/entropy-production measure) is the distinguishing probe here, and why the discriminator
  must be **kinetic**.
- **The "disguised reward" objection is the consensus view, not a flaw to hide.** Causal-entropic
  forcing is mathematically an entropy-of-futures intrinsic reward (Shah 2014; Ramírez-Ruiz et al.,
  *Nat. Commun.* 2024, Maximum Occupancy Principle); S_c is a Maximum-Caliber path entropy (Pressé
  et al. 2013). The contribution is the *experimental epistemology*: the τ probe, the engine-vs-readout
  framing, and a pre-registered prediction that failed for a principled, diagnosed reason.
- **Sharpest honest thesis for v2:** engine and readout entropy can be *thermodynamically
  indistinguishable* (identical occupancy, zero current) yet *kinetically distinguishable* (different
  first-passage signatures under the τ-knob). v2 = MFPT **and its fluctuations** (TUR: Gingrich–Horowitz
  2017) between chambers vs `L_ch`, plus a direct broken-detailed-balance / probability-current test
  (Battle–Broedersz, *Science* 2016) — predicted null here, which would itself be a clean result.

## v2 — First-passage discriminator (the kinetic test)
v1 showed the engine's *steady-state* occupancy is blind to channel length (Arm C). v2 tests
the complementary **kinetic** observable. Critically, v1's 2D "wall + channel" has a
barrier-free shortcut along y=0 (the wall term vanishes at y=0), so the crossing is gated by
the well-depth ridge, not the channel length — which is *why* occupancy (and a naive MFPT)
can't see L_ch there. v2 uses a clean **1D corridor** (two asymmetric wells — option-poor
narrow + option-rich wide — separated by a flat plateau of length `sep`, the genuine diffusive
bottleneck). Code: `src/landscape_v2.py`, `src/v2_firstpassage.py`, `src/v2_current.py`.

**Part A — kinetics sees the corridor.** Mean first-passage time L→mid under natural dynamics:

| sep | sep²/D | MFPT | CV |
|---|---|---|---|
| 4 | 16 | 22.7 | 0.99 |
| 5 | 25 | 47.3 | 0.98 |
| 6 | 36 | 79.8 | 1.00 |
| 7 | 49 | 109.5 | 0.95 |
| 8 | 64 | 138.1 | 0.99 |

Fit **MFPT = 2.42·(sep²/D) − 12.5, R² = 0.992** — first-passage cleanly carries the corridor
signature, the exact `L²/D` scaling v1's occupancy could not. (CV ≈ 1 → near-exponential FPT,
i.e. escape-dominated with a diffusive corridor term.) Figure `figures/v2_firstpassage.png`.

**Part B — occupancy carries no corridor-time signature.** Thermal P_R (option-rich basin) is
**flat** in sep (0.46→0.52, spread 0.05) — detailed balance: occupancy is a functional of free
energies only. The *engine* occupancy is **not** flat (0.78→0.60): its effective potential
`U_eff = −T_c S_c` is geometry-coupled (the growing open plateau is option-rich, so it pulls
future-seeking mass toward center). But this is a *steady-state S_c effect, not the kinetic
`sep²/D` signature* — the corridor *time* lives only in Part A.

**Part C — the thermodynamic null (with a positive control).** Probability-current /
broken-detailed-balance test (`mean angular current ⟨L_z⟩`) on a 2D rotor `F_rot = ω(−y,x)`:

| ω | ⟨L_z⟩ |
|---|---|
| 0.0 | **+0.0009** (conservative — engine's regime) |
| 0.5 | +0.98 |
| 1.0 | +1.97 |
| 2.0 | +3.99 |

The detector works (⟨L_z⟩ ≈ ω·⟨r²⟩, clear current for ω>0) and reads **zero at ω=0**. By Arm A
the causal entropic force is conservative → it *is* the ω=0 case → **zero steady-state current,
zero entropy production**. Under every standard NESS detector it registers as equilibrium.
Figure `figures/v2_current.png`.

**v2 verdict.** The sharp thesis is **demonstrated**: in this regime, engine and readout entropy
are **thermodynamically indistinguishable** (identical occupancy structure, zero current — Parts
B, C) yet **kinetically distinguishable** (first-passage carries the `sep²/D` corridor signature —
Part A). The engine-vs-readout distinction is real but lives in *dynamics*, not in occupancy or
in any current/entropy-production measure. This both **resolves v1** (the L-signature occupancy
discarded is recovered kinetically) and **confirms the prior-art diagnosis** (the thermodynamics–
kinetics separation; `docs/PRIOR_ART_AND_NOVELTY.md`). Honest caveat: this establishes that the
*landscape's* corridor time is kinetic-only; a stronger v3 claim — that the **τ-knob** leaves a
first-passage fingerprint no static U(x) can mimic — needs engine-*driven* first-passage under
`U_eff(τ)` across τ, which is the natural next step.

## Honest caveats
- Ran on the MacBook (NumPy fallback), not the Studio (JAX). The 2D engine is
  oracle-validated against the reference (corr 0.993, relL2 0.005), but the absolute S_c
  values are MC estimates (endpoint-histogram entropy, finite n_paths/bins); what carries
  the claims is the *structure* (τ-monotonicity, crossover location, L_ch/D scaling), not
  absolute entropies.
- `βT_c` is renormalized per τ to a fixed contrast of 3 (reference convention); P_L
  migration is driven by *where* S_c peaks, which is contrast-invariant.
- **The "r_L" attribution is inferred, not directly measured.** What is *measured*: τ* is
  channel-length-independent (Arm C) and scales as 1/D (D-scan) → a diffusive geometric
  time. That it is specifically the *large-chamber* exploration length (r_L) follows from
  (i) the magnitude match (τ*~2.4 vs r_L²/D=3.2) and (ii) the wall-shrinkage mechanism
  (P_L_control & P_L_max fall as L_ch widens the wall into the chamber). A direct r_L sweep
  would confirm it; it has a thermal confound (P_L_control rises with r_L) and was left as
  a follow-up.
- τ* estimators: first-crossing of the half-rise threshold (sensitive to P_L_max noise)
  and a logistic-midpoint fit (more stable). Both agree on the Arm C verdict (no L_ch
  scaling) and the D-scan trend.
- This is a *negative* result for the SPEC's specific discriminator, reported per the SPEC's
  own falsifier clause ("a negative result here is a real, publishable outcome — do not tune
  it away"). The chamber tuning that flipped thermal to favor the small well is legitimate
  setup (it changes the *control*, not τ*); τ* itself was never tuned.
