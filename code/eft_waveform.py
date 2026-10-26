#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tao Zhu (project additions).
# Adapted FTI code: Copyright (C) 2017 Walter Del Pozzo.
# Modified 2026-10-18; see LICENSE.
"""GR waveform with EFT phase corrections using the FTI method.

Formula references:
[1] Liu & Yunes, arXiv:2407.08929v2, Eqs. (24)-(26): cubic 5PN/6PN phase.
    https://arxiv.org/abs/2407.08929v2
[2] Sennett et al., arXiv:1912.09917v2, Sec. III B: quartic 8PN phase.
    https://arxiv.org/abs/1912.09917v2
[3] Mehta et al., Phys. Rev. D 107, 044020 (2023), Eqs. (7)-(15): FTI.
    https://arxiv.org/abs/2203.13937
[4] Bohe et al., arXiv:1611.03703, Eq. (2.8) and Appendix A, Eq. (A8).
    https://arxiv.org/abs/1611.03703
[5] LALSuite v7.26.1, lalsimulation/lib/LALSimInspiralTestingGRCorrections.c:
    Get22PeakAngFreq and XLALSimInspiralPhaseCorrectionsPhasing.
    https://git.ligo.org/lscsoft/lalsuite/-/blob/lalsuite-v7.26.1/lalsimulation/lib/LALSimInspiralTestingGRCorrections.c
[6] bilby_tgr 0.3, bilby_tgr/fti/source.py (SEOBNRv5HM_ROM branch):
    reference implementation for the mode-by-mode FTI workflow.
    Repository: https://git.ligo.org/lscsoft/bilby_tgr
    Release: https://pypi.org/project/bilby-tgr/0.3/

The n=10,12,16 additive EFT coefficients extend the standard FTI coefficient set. 
"""
import numpy as np

# G*M_sun/c^3 in seconds.
MTSUN_SI = 4.9254909476412675e-6
MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (5, 5)]
BETAS = ["beta1", "beta2", "beta3"]


def peak_angular_frequency(eta, chi):
    """Dimensionless 22-mode peak angular frequency (Bohé 2017, arXiv: 1611.03703, Eq. A8). Only used for aligned-spin systems and no precession."""
    return 0.5626787200433265 + (
        -0.08706198756945482 + 0.0017434519312586804 * chi
    ) * np.log(
        10.26207326082448
        - chi * (7.629921628648589 - 72.75949266353584 * (-0.25 + eta))
        - 62.353217004599784 * (-0.25 + eta)
    )


def phase_basis(f, p, fmin, fref, F=1.0, cycles=1.0, strict_peak=True):
    """Return unwrapped LAL-phase derivatives."""
    M = (p["mass_1"] + p["mass_2"]) * MTSUN_SI
    eta = p["mass_1"] * p["mass_2"] / (p["mass_1"] + p["mass_2"]) ** 2
    delta = (p["mass_1"] - p["mass_2"]) / (p["mass_1"] + p["mass_2"])
    # Peak-fit spin: [4], Eq. (2.8).
    chi_fit = (p["chi_1z"] + p["chi_2z"]) / 2 + delta * (p["chi_1z"] - p["chi_2z"]) / (
        2 * (1 - 2 * eta)
    )
    peak = peak_angular_frequency(eta, chi_fit) / (2 * np.pi * M)
    if strict_peak and peak >= f[-1]:
        raise ValueError(
            f"fmax={f[-1]:g} does not contain f22_peak={peak:g}; increase fmax with PSD coverage"
        )
    valid = f >= fmin / 2
    x = f[valid]
    ip = min(int(np.searchsorted(x, peak)), len(x) - 1)
    vw = (np.pi * M * F * x[ip]) ** (1 / 3)
    # Fermi-window width following LALSuite [5].
    width = cycles * np.pi * vw**6 / (50 * 3 / (128 * eta))
    # 5PN/6PN coefficients: [1]; 8PN: [2]. Columns follow paper beta1,beta2,beta3.
    coeffs = {
        10: [0.0, -1872.0, 0.0],
        12: [549360 / 294, -45 * (43683 + 12908 * eta) / 294, 0.0],
        16: [0.0, 0.0, (234240 - 522240 * eta) / 11],
    }
    result = {}
    for m in sorted({m for _, m in MODES}):
        ir = min(int(np.searchsorted(x, fref * m / 2)), len(x) - 1)
        if ir >= len(x):
            raise ValueError("mode reference frequency outside the waveform grid")
        v = (2 * np.pi * M * x / m) ** (1 / 3)
        # FTI tapers the second phase derivative [3,5].
        w = 1 / (1 + np.exp(np.clip((v - vw) / width, -700, 700)))
        d2 = np.zeros((3, len(x)))
        for k, c in coeffs.items():
            power = (k - 5) / 3
            # LAL sign convention: Q_bare = -delta_Psi_lit.
            d2 += (
                -3
                / (128 * eta)
                * (m / 2)
                * np.asarray(c)[:, None]
                * v[None, :] ** (k - 5)
                * power
                * (power - 1)
                / x[None, :] ** 2
            )

        def integrate(y):
            out = np.column_stack(
                [
                    np.zeros(3),
                    np.cumsum((y[:, 1:] + y[:, :-1]) * np.diff(x) / 2, axis=1),
                ]
            )
            return out - out[:, ir, None]

        # Double integration [3,5]: Q(f_ref,m)=0 and Q'(f_peak)=0.
        first = integrate(d2 * w)
        phase = integrate(first) - first[:, ip, None] * (x - x[ir])
        g = np.zeros((3, len(f)))
        g[:, valid] = phase
        g[:, f < fmin] = 0.0
        result[m] = g
    return result, peak


def gr_waveform(f, p, fmin, fref, modes=None):
    import bilby

    return bilby.gw.source.lal_binary_black_hole(
        f,
        mass_1=p["mass_1"],
        mass_2=p["mass_2"],
        luminosity_distance=p["luminosity_distance"],
        a_1=abs(p["chi_1z"]),
        a_2=abs(p["chi_2z"]),
        tilt_1=0.0 if p["chi_1z"] >= 0 else np.pi,
        tilt_2=0.0 if p["chi_2z"] >= 0 else np.pi,
        phi_12=0.0,
        phi_jl=0.0,
        theta_jn=p["iota"],
        phase=p["phase"],
        waveform_approximant="SEOBNRv5HM_ROM",
        minimum_frequency=fmin,
        reference_frequency=fref,
        mode_array=[(l, -m) for l, m in (modes or MODES)],
    )


def waveform_eft(
    frequency_array,
    mass_1,
    mass_2,
    luminosity_distance,
    a_1,
    tilt_1,
    phi_12,
    a_2,
    tilt_2,
    phi_jl,
    theta_jn,
    phase,
    beta1=0.0,
    beta2=0.0,
    beta3=0.0,
    **kwargs,
):
    import bilby

    F = kwargs.pop("f_window_div_f_peak", 1.0)
    cycles = kwargs.pop("num_cycles", 1.0)
    fmin, fref = kwargs.get("minimum_frequency", 20.0), kwargs.get(
        "reference_frequency", 20.0
    )
    p = dict(
        mass_1=mass_1,
        mass_2=mass_2,
        chi_1z=a_1 * np.cos(tilt_1),
        chi_2z=a_2 * np.cos(tilt_2),
    )
    # Convert archived beta3_code = 3*beta3_paper.
    beta = np.array([beta1, beta2, beta3 / 3.0])
    # Retain the archived Bayesian peak-index clamp.
    phases = (
        phase_basis(frequency_array, p, fmin, fref, F, cycles, strict_peak=False)[0]
        if np.any(beta)
        else None
    )
    result = {k: np.zeros(len(frequency_array), complex) for k in ("plus", "cross")}
    for l, m in MODES:
        mode = bilby.gw.source.lal_binary_black_hole(
            frequency_array,
            mass_1=mass_1,
            mass_2=mass_2,
            luminosity_distance=luminosity_distance,
            a_1=a_1,
            a_2=a_2,
            tilt_1=tilt_1,
            tilt_2=tilt_2,
            phi_12=phi_12,
            phi_jl=phi_jl,
            theta_jn=theta_jn,
            phase=phase,
            mode_array=[(l, -m)],
            **kwargs,
        )
        correction = (
            np.exp(1j * np.sum(beta[:, None] * phases[m], axis=0))
            if phases is not None
            else 1.0
        )
        for k in result:
            result[k] += mode[k] * correction
    return result
