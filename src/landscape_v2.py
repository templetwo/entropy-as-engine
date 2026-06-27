"""
landscape_v2.py
===============
Corridor landscapes for the v2 first-passage discriminator.

v1's 2D "wall pierced by a channel" has a barrier-free shortcut along y=0 (the wall
term B0*wall_x(x)*gap_y(y) vanishes at y=0), so the inter-well crossing is gated by
the well-depth ridge, NOT the channel length. That is why neither the steady-state
occupancy (Arm C) nor a naive MFPT sees L_ch there. To honestly test tau* ∝ L²/D we
need a geometry where the corridor length IS the diffusive bottleneck.

Corridor1D: two asymmetric wells (option-poor narrow + option-rich wide, as in v1)
separated by a flat plateau whose length is set by the well separation `sep`. Crossing
requires diffusing the plateau (∝ sep²/D), while the equilibrium occupancy ratio depends
only on the well free energies (∝ exp(βΔF)) and is independent of `sep`. That separation
of "occupancy blind to L / kinetics ∝ L²" is exactly the v2 thesis.

Rotor2D: a 2D landscape + an optional NON-conservative rotational force, used as a
positive control for the probability-current / broken-detailed-balance test (a real NESS
must show a nonzero steady-state current; the conservative engine force must not).
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class Corridor1D:
    """Two Gaussian wells at ∓sep/2 with a flat (U≈0) plateau between them.

      U(x) = -d_S exp(-(x-cL)²/2σ_S²)   (left,  narrow/deep  -> option-poor)
             -d_L exp(-(x-cR)²/2σ_L²)   (right, wide/shallow -> option-rich)
      cL = -sep/2,  cR = +sep/2.

    `sep` is the v2 knob (the diffusive corridor length). Domain is sized to sep so the
    plateau is genuinely flat and reflecting walls sit well outside both wells.
    """
    sep: float = 5.0
    d_S: float = 4.0
    sig_S: float = 0.6      # narrow, deep   -> option-poor trap
    d_L: float = 3.0
    sig_L: float = 1.6      # wide, shallow  -> option-rich basin
    pad: float = 3.0

    @property
    def cL(self): return -self.sep / 2.0
    @property
    def cR(self): return +self.sep / 2.0
    @property
    def xl(self): return self.cL - self.pad
    @property
    def xr(self): return self.cR + self.pad

    def U(self, x):
        a = -self.d_S * np.exp(-((x - self.cL) ** 2) / (2 * self.sig_S ** 2))
        b = -self.d_L * np.exp(-((x - self.cR) ** 2) / (2 * self.sig_L ** 2))
        return a + b

    def dU(self, x):
        a = -self.d_S * np.exp(-((x - self.cL) ** 2) / (2 * self.sig_S ** 2)) * (-(x - self.cL) / self.sig_S ** 2)
        b = -self.d_L * np.exp(-((x - self.cR) ** 2) / (2 * self.sig_L ** 2)) * (-(x - self.cR) / self.sig_L ** 2)
        return a + b

    def reflect(self, x):
        x = np.where(x < self.xl, 2 * self.xl - x, x)
        x = np.where(x > self.xr, 2 * self.xr - x, x)
        return np.clip(x, self.xl, self.xr)


@dataclass
class Rotor2D:
    """A single 2D well plus an optional solenoidal (curl) force F_rot = omega*(-y, x).
    omega=0 -> conservative (equilibrium, zero current). omega>0 -> genuine NESS with a
    steady circulating probability current. Positive control for the current detector."""
    k: float = 1.0          # harmonic stiffness of the confining well
    omega: float = 0.0      # rotational drive strength (0 => conservative)
    box: float = 3.0

    def drift(self, p):
        x, y = p[..., 0], p[..., 1]
        fx = -self.k * x - self.omega * y
        fy = -self.k * y + self.omega * x
        return np.stack([fx, fy], axis=-1)

    def reflect(self, p):
        return np.clip(p, -self.box, self.box)
