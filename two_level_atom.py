#!/usr/bin/env python3
"""
Decay of a two-state atom coupled to "photons"
==============================================

Python 3 implementation of the second problem in

    Eva Obeed, "Numerical Modelling of Some Quantum Systems",
    Master thesis, NTNU, Trondheim, May 2016   (sections 4.2 and 6.2).

In the sector with one excitation the Hamiltonian is the (N+2) x (N+2)
"arrowhead" matrix of eq. (6.9),

        | Ea  g   g  ...  g  |
        | g   E_0 0  ...  0  |
    H = | g   0   E_1 ... 0  |        E_n evenly spaced on [0, Emax]
        | :               :  |
        | g   0   0  ... E_N |

State 0 is the excited atom with no photon; state n+1 is the atom in its
ground state plus one photon of energy E_n.

The Schroedinger equation i d(psi)/dt = H psi, psi(0) = (1, 0, ..., 0), is
solved by full diagonalization, H = S diag(E) S^T, giving the survival
amplitude (eq. 4.2)

    M0(t) = <psi0|psi(t)> = sum_n exp(-i E_n t) |S[0, n]|^2 ,
    p0(t) = |M0(t)|^2 .

It is compared with the Fermi Golden Rule, p0(t) = exp(-Gamma0 t), with
Gamma0 = 2 pi g^2 rho(Ea), where rho = N / Emax is the density of states.
"""

import os

import numpy as np
from scipy.linalg import eigh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TwoLevelAtom:
    """Two-level atom coupled to N+1 equally spaced photon modes.

    Parameters
    ----------
    N     : number of photon modes is N+1
    lamda : coupling strength; the coupling per mode is
            g = lamda * sqrt(dE), dE = Emax / N, so that the decay rate
            Gamma0 = 2 pi lamda^2 does not depend on N or Emax
    Ea    : energy of the excited atomic state
    Emax  : upper edge of the photon spectrum (default 8 * Ea)
    noise : random perturbation of the photon energies, in units of dE
            (0 <= noise < 1)
    """

    def __init__(self, N=1000, lamda=0.1, Ea=1.0, Emax=None, noise=0.0,
                 rng=None):
        self.N = N
        self.Ea = Ea
        self.Emax = 8 * Ea if Emax is None else Emax
        self.dE = self.Emax / N                       # inverse density of states
        self.Eph = np.linspace(0, self.Emax, N + 1)   # photon spectrum
        self.g = lamda * np.sqrt(self.dE)             # coupling per mode
        if noise:
            rng = np.random.default_rng() if rng is None else rng
            self.Eph = self.Eph + noise * (rng.random(N + 1) - 0.5) * self.dE
        self._eig = None

    # -- Hamiltonian ---------------------------------------------------------
    def H(self, psi):
        """Action of H on a state vector (linear operator form)."""
        psi = np.asarray(psi)
        out = np.empty_like(psi, dtype=np.result_type(psi, float))
        out[0] = self.Ea * psi[0] + self.g * np.sum(psi[1:])
        out[1:] = self.Eph * psi[1:] + self.g * psi[0]
        return out

    def matrix(self):
        """Dense matrix of eq. (6.9)."""
        H = np.diag(np.concatenate(([self.Ea], self.Eph)))
        H[0, 1:] = self.g
        H[1:, 0] = self.g
        return H

    @property
    def golden_rule_rate(self):
        """Gamma0 = 2 pi g^2 rho(Ea) = 2 pi g^2 / dE   (eq. 6.13)."""
        return 2 * np.pi * self.g ** 2 / self.dE

    # -- time evolution ----------------------------------------------------------
    def diagonalize(self):
        if self._eig is None:
            self._eig = eigh(self.matrix())
        return self._eig

    def amplitudes(self, t):
        """Full wave function psi(t) for each time in t.
        Returns a complex array of shape (len(t), N+2)."""
        E, S = self.diagonalize()
        t = np.atleast_1d(t)
        phases = np.exp(-1j * t[:, None] * E[None, :])    # broadcast: Nt x (N+2)
        return (phases * S[0, :]) @ S.T

    def survival_probability(self, t):
        """p0(t) = |<psi0|psi(t)>|^2   (eqs. 4.2 - 4.3)."""
        E, S = self.diagonalize()
        t = np.atleast_1d(t)
        Et = t[:, None] * E[None, :]
        S02 = S[0, :] ** 2
        M0R = np.cos(Et) @ S02
        M0I = np.sin(Et) @ S02
        return M0R ** 2 + M0I ** 2


# ---------------------------------------------------------------------------
# The three numerical experiments of section 6.2
# ---------------------------------------------------------------------------

def golden_rule_comparison(outdir=".", Nt=400):
    """Figure 6.1: numerical p0(t) against exp(-Gamma0 t)."""
    atom = TwoLevelAtom()
    G0 = atom.golden_rule_rate
    t = np.linspace(0, 5 / G0, Nt)
    p0 = atom.survival_probability(t)

    fig, ax = plt.subplots(figsize=(6, 3.7))
    ax.plot(G0 * t, np.exp(-G0 * t), "b", lw=2.5, label="Golden rule")
    ax.plot(G0 * t, p0, "y", lw=1, label="Numerical")
    ax.set(xlim=(0, 5), xlabel=r"$\Gamma_0\,t$", ylabel=r"$p_0(t)$",
           title="Numerical rate and Golden rule prediction")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(outdir, "TwoLevelAtomicDecay0.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    return t, p0, G0


def short_time_behavior(outdir=".", Nt=200):
    """Figure 6.2: quadratic (non-exponential) decay at short times, for
    several Emax at constant Gamma0."""
    fig, ax = plt.subplots(figsize=(6, 3.7))
    results = {}
    for Emax, style in [(2, "b-."), (4, "g--"), (8, "r")]:
        atom = TwoLevelAtom(Emax=Emax)
        G0 = atom.golden_rule_rate
        t = np.linspace(0, 0.12 / G0, Nt)
        p0 = atom.survival_probability(t)
        ax.plot(G0 * t, p0, style, label=r"$E_{\max} = %d$" % Emax)
        results[Emax] = (t, p0, atom)
    ax.set(ylim=(0.94, 1.002), xlabel=r"$\Gamma_0\,t$", ylabel=r"$p_0(t)$",
           title="Short time behavior")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(outdir, "TwoLevelAtomicDecay1.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    return results


def long_time_revival(outdir=".", Nt=1500, Ns=(500, 1250, 2000)):
    """Figure 6.3: revivals of the excited state at long times. The revival
    time is 2 pi / dE, i.e. proportional to N."""
    fig, ax = plt.subplots(figsize=(6, 3.7))
    results = {}
    for N, style in zip(Ns, ["b", "g--", "r-."]):
        atom = TwoLevelAtom(N=N)
        G0 = atom.golden_rule_rate
        t = np.linspace(0, 140 / G0, Nt)
        p0 = atom.survival_probability(t)
        ax.semilogy(G0 * t, p0, style, lw=1, label="$N=%d$" % N)
        results[N] = (t, p0, atom)
    ax.set(xlim=(0, 140), xlabel=r"$\Gamma_0\,t$", ylabel=r"$p_0(t)$",
           title="Long time revivals")
    ax.legend(loc="upper right", framealpha=0.2)
    fig.savefig(os.path.join(outdir, "TwoLevelAtomicDecay2.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    return results


def main(outdir="figures"):
    os.makedirs(outdir, exist_ok=True)

    t, p0, G0 = golden_rule_comparison(outdir)
    print("Golden rule rate Gamma0 = %.6f" % G0)
    sel = (G0 * t > 0.5) & (G0 * t < 4)
    fit = -np.polyfit(t[sel], np.log(p0[sel]), 1)[0]
    print("Fitted numerical rate   = %.6f  (0.5 < Gamma0 t < 4)" % fit)

    print("\nShort times: 1 - p0(t) -> (sum_n g_n^2) t^2")
    for Emax, (t, p0, atom) in short_time_behavior(outdir).items():
        coeff = (1 - p0[5]) / t[5] ** 2
        exact = atom.g ** 2 * (atom.N + 1)
        print("  Emax=%d: numerical %.6f, predicted %.6f" % (Emax, coeff, exact))

    print("\nLong times: first revival near Gamma0 * 2 pi / dE")
    for N, (t, p0, atom) in long_time_revival(outdir).items():
        G0 = atom.golden_rule_rate
        late = G0 * t > 20
        t_rev = t[late][np.argmax(p0[late])]
        print("  N=%4d: revival at Gamma0 t = %6.1f, predicted %6.1f"
              % (N, G0 * t_rev, G0 * 2 * np.pi / atom.dE))
    print("\nFigures written to", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
