# Entropy as Engine

**Can entropy drive a system, or only describe one?**

This repo runs a single falsifiable experiment to tell two views of entropy apart:

- **Readout entropy** — entropy is a thermometer. A system maximizes its own entropy by relaxing to equilibrium, and stops. The steady state is fixed (the Boltzmann distribution). There is no knob you can push.
- **Engine entropy** — entropy is a prime mover. Using the *causal entropic force* (Wissner-Gross & Freer, *PRL* 2013), a future-entropy gradient drives a system into organized, anticipatory behavior with **no reward function**.

The two are separable by one device: a **horizon knob** τ. If you can change how a system organizes itself purely by changing how far ahead it looks, entropy is doing work, not just being measured. Thermal relaxation has no τ. The causal entropic force does.

## The minimal prediction

> The engine's steady-state organization is a **monotone function of the horizon τ**, with a **crossover τ\*** set by the landscape geometry. Readout entropy has no τ and predicts a fixed distribution.

And the discriminating test, against the obvious objection that "the entropic force is just a reward in disguise":

> **τ\* ∝ L_channel² / D** (the diffusive traversal time). A smuggled constant reward would pull at all horizons and could not reproduce this scaling.

## What's here

A particle in a potential landscape under overdamped Langevin dynamics, two conditions:

1. **Control** — pure thermal relaxation to `p_eq ∝ exp(−U/kT)`. Fixed, no horizon.
2. **Engine** — driven by the causal entropic force `F = T_c ∇S_c(x, τ)`, where `S_c` is the entropy over futures reachable within horizon τ. Steady state `p_engine ∝ exp(T_c S_c / kT)` (for the overdamped case, where the force is conservative — a property the experiment verifies rather than assumes).

## Status

| Phase | What | State |
|---|---|---|
| **Phase 0** | 1D dumbbell, mechanism check | ✅ validated + **reproduced** (MacBook) — engine ≠ readout (KL 0.79–1.14), horizon knob confirmed (mass 0.483 → 0.668 as τ grows; control fixed 0.527) |
| **Arm A** | Conservativeness check (drive a particle, confirm the analytic steady state) | ✅ **PASS** — KL(driven‖analytic) 0.0002, overlap 0.993; analytic shortcut licensed |
| **Arm B** | 2D chamber–channel–chamber, sharp τ\* crossover | ✅ **horizon knob confirmed** — engine P_L 0.51 → 0.91 monotone vs flat control 0.371 |
| **Arm C** | τ\* tracks geometry (`τ\* ∝ L_ch²/D`) — the make-or-break test | ❌ **FALSIFIED** — τ\* is flat in L_ch (R²ₒᵣᵢgᵢₙ = −17.7). A D-scan shows τ\* ∝ 1/D (R²=0.97): τ\* is a diffusive geometric time set by **chamber** exploration (r_L), not **channel** length. See `RESULTS.md`. |
| **v2** | First-passage discriminator (kinetic) | ✅ **thesis demonstrated** — MFPT ∝ sep²/D (R²=0.992) recovers the corridor signature occupancy is blind to; current test ⟨L_z⟩≈0 for the conservative engine. Engine vs readout: thermodynamically indistinguishable, kinetically distinguishable. |
| Arm D | Free Energy Principle (dual engine) | 🔜 future |

> **Outcome (see [RESULTS.md](RESULTS.md)).** The horizon knob is real (Phase 0, Arm B) and
> the engine's organization obeys a genuine diffusive law (τ\* ∝ 1/D). The steady-state
> discriminator τ\* ∝ L_ch²/D **failed** (Arm C) — a textbook consequence of detailed balance:
> occupancy is blind to path lengths, which live only in kinetics. **v2 confirms the resolution:**
> in a clean corridor, first-passage time **does** carry the corridor signature (MFPT ∝ sep²/D,
> R²=0.992) while occupancy stays flat and the conservative engine force shows **zero probability
> current** (⟨L_z⟩≈0 vs a rotational positive control). The sharp, defensible thesis: **engine and
> readout entropy can be thermodynamically indistinguishable (identical occupancy, zero current)
> yet kinetically distinguishable (first-passage under the τ-knob).** Prior-art context and honest
> novelty accounting: [docs/PRIOR_ART_AND_NOVELTY.md](docs/PRIOR_ART_AND_NOVELTY.md).

### Phase 0 result

Two valleys: a deep narrow well (option-poor) and a wide shallow well (option-rich). Thermal relaxation dives into the deep well and stays. The entropy drive *refuses* the deep well (a narrow trap has few open futures) and, as the horizon grows, slides into the wide well — monotonically. The control sheet is flat in τ; the engine surface tilts and migrates. That tilt is the knob, and the knob is the thesis. (See `figures/`.)

## Quick start

```bash
pip install -r requirements.txt
python -m src.phase0_validated_1d        # reproduce Phase 0 + regenerate the 3D overlay figure
python -m src.run --arm A                 # conservativeness check
python -m src.run --arm B                 # 2D crossover
python -m src.run --arm C                 # geometry-scaling discriminator
```

## How to read the result

This experiment is built to be **falsifiable**. If the horizon knob does nothing, or τ\* does not track `L_ch²/D`, the engine-as-prime-mover claim fails — and that negative result gets reported. The point is not to confirm a belief; it is to find out which view of entropy survives a test that could have killed it.

## Why it matters

The arrow of time *is* entropy increase, so "is entropy fundamental?" and "is time fundamental?" are the same question. The deep-physics version of this fork runs from Boltzmann's H-theorem (the clean statement of readout-entropy as one-way drift) through the block-universe / thermal-time view (Rovelli — entropy as readout of coarse-graining) versus time-as-real (Smolin). This experiment doesn't settle that debate. It does something smaller and checkable: it asks whether an entropy gradient *alone* can generate sustained, horizon-dependent organization, and pins the answer to a number you can measure.

## Details

Full experimental specification, parameters, success/falsification criteria, and executor instructions: **[SPEC.md](SPEC.md)**.

## Related work & honest novelty

A prior-art and novelty assessment is in **[docs/PRIOR_ART_AND_NOVELTY.md](docs/PRIOR_ART_AND_NOVELTY.md)**.
Short version of where this sits in the literature:

- **The mechanism is not new.** The causal entropic force (Wissner-Gross & Freer, *PRL* 2013)
  and its option-rich-preference behavior are established, and the "is this just reward
  shaping?" worry is the consensus view: causal-entropic forcing is mathematically an
  *entropy-of-futures intrinsic reward* (Shah 2014; Ramírez-Ruiz et al., *Nat. Commun.* 2024,
  "Maximum Occupancy Principle"). S_c is best framed as a **Maximum-Caliber path entropy**
  (Pressé et al. 2013; Ghosh et al. 2020).
- **The v1 Arm C failure is textbook, and that's the point.** That a *steady-state*
  occupancy can't see channel-traversal time is the detailed-balance / thermodynamics–kinetics
  separation: equilibrium ∝ exp(−βU_eff) is provably blind to barriers and path lengths, which
  live only in *kinetic* observables (Hänggi–Talkner–Borkovec, *RMP* 1990; Gardiner; van Kampen).
  The expected `L²/D` scaling is canonical diffusive first-passage — it appears in the **MFPT**,
  not in occupancy.
- **The genuinely defensible contribution** is the experimental epistemology: the **τ horizon
  knob** as a controlled dial, the **engine-vs-readout** framing as a testable question, and a
  *pre-registered prediction that failed for a principled reason* with a correct diagnosis. The
  sharpest honest thesis: in the conservative, position-only overdamped regime, "engine entropy"
  has zero steady-state current / zero entropy production — it is **thermodynamically
  indistinguishable** from equilibrium relaxation and distinguishable, if at all, only
  **kinetically**. That is what the v2 first-passage discriminator is built to test.

## Provenance

Part of [The Temple of Two](https://thetempleoftwo.com) · [github.com/templetwo](https://github.com/templetwo). Designed collaboratively by Anthony Vasquez and Claude. Background research chain and persistent record live in the Sovereign Stack (entropy-as-substrate thread).

## License

Apache-2.0. See [LICENSE](LICENSE).
