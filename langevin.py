#!/usr/bin/env python3
"""
Langevin description of damped, randomly forced oscillators
===========================================================

Python 3 implementation of the classical part of

    Eva Obeed, "Numerical Modelling of Some Quantum Systems",
    Master thesis, NTNU, Trondheim, May 2016
    (sections 5.5, 5.6 and 6.3 - 6.5).

The linear Langevin equations are written as a first order system

    dz/dt + D z = F(t),     z = (q, p),

with the dynamical matrix D. For one oscillator, H = (p^2 + q^2)/2,

    D = [[0, -1], [1, gamma]],      F = (0, f(t)),
    <f(t)> = 0,  <f(t) f(u)> = K delta(t - u).

With the eigen-decomposition D = R diag(lam) L, L = R^-1:

    mean:        z(t) = R exp(-lam t) L z0                      (eq. 6.20)
    covariance:  C_mn(t) = K sum_kl R_mk L_k2 R_nl L_l2
                           (1 - exp(-(lam_k+lam_l) t)) / (lam_k+lam_l)
                                                                (eqs. 5.82-83)

The thesis notes that this part "did not reach a conclusive stage"; the
stochastic simulation and the chain below complete it in the direction the
text describes.
"""

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class LinearLangevin:
    """dz/dt + D z = F(t), with white noise of covariance G delta(t-u)
    (G is a matrix: <F_m(t) F_n(u)> = G_mn delta(t-u))."""

    def __init__(self, D, G=None):
        self.D = np.asarray(D, dtype=float)
        self.n = self.D.shape[0]
        self.G = np.zeros((self.n, self.n)) if G is None else np.asarray(G, float)
        self.evals, self.R = np.linalg.eig(self.D)
        self.L = np.linalg.inv(self.R)

    def propagator(self, t):
        """exp(-t D)."""
        return np.real((self.R * np.exp(-t * self.evals)) @ self.L)

    def mean_path(self, z0, tvals):
        """Deterministic (noise-averaged) solution at the times tvals."""
        g0 = self.L @ np.asarray(z0, dtype=complex)
        gt = np.exp(-np.outer(tvals, self.evals)) * g0      # Nt x n
        return np.real(gt @ self.R.T)

    def covariance(self, t):
        """Equal-time covariance C(t) of the fluctuations around the mean.
        t = numpy.inf gives the stationary (thermal) covariance."""
        A = self.L @ self.G @ self.L.T                      # noise in eigenbasis
        s = self.evals[:, None] + self.evals[None, :]
        decay = 0.0 if np.isinf(t) else np.exp(-s * t)
        return np.real(self.R @ (A * (1 - decay) / s) @ self.R.T)

    def simulate(self, z0, tmax, dt=0.01, npaths=1000, rng=None):
        """Stochastic simulation of an ensemble of paths.

        Because the system is linear, one time step can be taken exactly:
            z(t+dt) = exp(-dt D) z(t) + xi,
        with xi Gaussian of covariance C(dt). There is no time step error,
        only sampling error from the finite number of paths.
        Returns (tvals, z) with z of shape (nsteps+1, npaths, n)."""
        rng = np.random.default_rng() if rng is None else rng
        nsteps = int(round(tmax / dt))
        U = self.propagator(dt)
        w, V = np.linalg.eigh(self.covariance(dt))          # C(dt) = B B^T
        B = V * np.sqrt(np.clip(w, 0, None))
        z = np.empty((nsteps + 1, npaths, self.n))
        z[0] = z0
        for k in range(nsteps):
            xi = rng.standard_normal((npaths, self.n))
            z[k + 1] = z[k] @ U.T + xi @ B.T
        return np.linspace(0, nsteps * dt, nsteps + 1), z


def damped_oscillator(gamma=0.1, K=0.0):
    """One-dimensional oscillator H = (p^2 + q^2)/2 with damping gamma and
    random force strength K (eq. 6.19)."""
    D = np.array([[0.0, -1.0], [1.0, gamma]])
    G = np.array([[0.0, 0.0], [0.0, K]])
    return LinearLangevin(D, G)


def analytic_propagator(t, gamma):
    """Closed form of exp(-t D) for the damped oscillator (eq. 6.21a),
    valid for gamma < 2."""
    w = np.sqrt(1 - gamma ** 2 / 4)
    kappa = gamma / (2 * w)
    c, s = np.cos(w * t), np.sin(w * t)
    return np.exp(-gamma * t / 2) * np.array([[c + kappa * s, s / w],
                                              [-s / w, c - kappa * s]])


def harmonic_chain(m, K, Q, gamma=None, noise=None):
    """Linear chain of d+1 oscillators (section 5.6),

        H = sum_k [ p_k^2 / (2 m_k) + K_k y_k^2 / 2 ]
            + sum_k Q_k (y_{k+1} - y_k)^2 / 2 .

    m, K : length d+1 (masses, on-site spring constants)
    Q    : length d (nearest neighbour couplings)
    gamma, noise : optional length d+1 damping rates and random force
                   strengths per site (e.g. nonzero only at site 0, eq. 5.68)

    Returns (system, omega): the LinearLangevin system for z = (y, p) and
    the normal mode frequencies of the undamped chain.
    """
    m, K, Q = (np.asarray(x, dtype=float) for x in (m, K, Q))
    n = m.size
    Kmat = np.diag(K)
    Kmat[:-1, :-1] += np.diag(Q)
    Kmat[1:, 1:] += np.diag(Q)
    Kmat -= np.diag(Q, 1) + np.diag(Q, -1)                  # eq. (5.87)
    Minv = np.diag(1 / m)
    gamma = np.zeros(n) if gamma is None else np.asarray(gamma, float)
    noise = np.zeros(n) if noise is None else np.asarray(noise, float)

    D = np.block([[np.zeros((n, n)), -Minv], [Kmat, np.diag(gamma)]])
    G = np.block([[np.zeros((n, n)), np.zeros((n, n))],
                  [np.zeros((n, n)), np.diag(noise)]])
    # Normal modes: mass-weighted matrix K_jk / sqrt(m_j m_k)   (eq. 5.70)
    omega2 = np.linalg.eigvalsh(Kmat / np.sqrt(np.outer(m, m)))
    return LinearLangevin(D, G), np.sqrt(np.clip(omega2, 0, None))


def random_parameters(mean, spread, size, rng=None):
    """Z = Zbar + dZ * zeta, zeta uniform on [-1/2, 1/2]   (eq. 5.88)."""
    rng = np.random.default_rng() if rng is None else rng
    return mean + spread * (rng.random(size) - 0.5)


def main(outdir="figures"):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(2016)

    # -- Section 6.5: damped oscillator by eigen-decomposition --------------
    gamma = 0.1
    osc = damped_oscillator(gamma)
    tvals = np.linspace(0, 20, 1001)
    z = osc.mean_path([0, 1], tvals)
    err = np.abs(osc.propagator(7.3) - analytic_propagator(7.3, gamma)).max()
    print("Damped oscillator: |exp(-tD) numeric - eq.(6.21)| = %.2e" % err)

    # -- Section 5.5.2: random force, covariance of (q, p) -------------------
    K = 0.2
    osc = damped_oscillator(gamma, K)
    ts, paths = osc.simulate([0, 1], tmax=60, dt=0.05, npaths=4000, rng=rng)
    print("\nRandomly forced oscillator, gamma=%.2f, K=%.2f" % (gamma, K))
    print("  stationary <q^2> = <p^2> = K/(2 gamma) = %.4f" % (K / (2 * gamma)))
    print("  formula   C(inf) =\n", osc.covariance(np.inf))
    print("  simulated C(60)  =\n", np.cov(paths[-1].T))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(tvals, z[:, 0], label="$q(t)$")
    axes[0].plot(tvals, z[:, 1], label="$p(t)$")
    axes[0].set(xlabel="Time $t$", ylabel="Magnitude",
                title=r"Damped oscillator, $\gamma = %.1f$" % gamma)
    axes[0].legend()
    var_q = paths[:, :, 0].var(axis=1)
    axes[1].plot(ts, var_q, lw=1, label="simulation (4000 paths)")
    axes[1].plot(ts[::50], [osc.covariance(t)[0, 0] for t in ts[::50]], "k--",
                 label="eq. (5.82)")
    axes[1].set(xlabel="Time $t$", ylabel=r"$\langle q^2\rangle - \langle q\rangle^2$",
                title="Fluctuations from the random force")
    axes[1].legend()
    fig.savefig(os.path.join(outdir, "LangevinOscillator.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    # -- Section 5.6: random chain, damped and driven at site 0 -------------
    d = 8
    m = random_parameters(1.0, 0.5, d + 1, rng)
    Kk = random_parameters(1.0, 0.5, d + 1, rng)
    Q = random_parameters(1.0, 0.5, d, rng)
    g = np.zeros(d + 1); g[0] = 0.3
    f = np.zeros(d + 1); f[0] = 0.3
    # damping term is gamma * p / m at site 0 (force -m gamma ydot, eq. 5.68)
    chain, omega = harmonic_chain(m, Kk, Q, gamma=g / m, noise=f)
    C = chain.covariance(np.inf)
    T = f[0] / (2 * g[0])
    print("\nChain of %d oscillators, bath on site 0 only" % (d + 1))
    print("  normal mode frequencies:", np.round(omega, 4))
    print("  bath temperature K/(2 gamma)          = %.4f" % T)
    print("  stationary <p_k^2>/m_k (equipartition):")
    print("  ", np.round(np.diag(C)[d + 1:] / m, 4))
    print("\nFigure written to", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
