#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tao Zhu (project additions).
"""Calculate Fisher bounds for 92 events using GR medians and PSD files."""
import csv
import os
import numpy as np
import bilby
import lal
from scipy.linalg import svd
from eft_waveform import BETAS, MODES, gr_waveform, phase_basis

# Change these settings before running the script.
EVENT_FILE = "./data/events.txt"
PSD_ROOT = "./psds/"  # Keep the final / in folder paths.
OUTPUT_DIR = "./output/"

RG_SUN_KM = lal.MTSUN_SI * lal.C_SI / 1000
Z90 = 1.6448536269514722  # Factor 1.645 for the 90% limit on |beta|.
# Frequency grid, derivative step and SVD cutoff used in the paper.
DF = 0.125  # Frequency spacing in Hz.
FMAX = 4095.75  # Highest grid frequency in Hz.
STEP = 1e-4
RCOND = 1e-10
CUTOFF_GUARD_HZ = 2.0
GR_PARAMETERS = (
    "mass_1", "mass_2", "chi_1z", "chi_2z", "luminosity_distance",
    "iota", "ra", "dec", "psi", "geocent_time", "phase",
)

def read_events(path):
    """Read the event parameter table as a list of dictionaries."""
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))

def load_psds(row, psd_root):
    """Read frequency (Hz) and one-sided PSD from each detector TXT file."""
    psds = {}
    for detector in row["pe_detectors"].split(","):
        path = psd_root + f"{row['event']}_psd_{detector}.txt"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing PSD: {path}. See README for PSD preparation.")
        frequency, values = np.loadtxt(path, unpack=True)
        keep = np.isfinite(frequency) & np.isfinite(values) & (values > 0)
        psds[detector] = dict(frequency=frequency[keep], psd=values[keep])
    row["_psds"] = psds
    return row

def geometry(ifo, p):
    """Return the plus/cross antenna factors and detector time delay."""
    args = p["ra"], p["dec"], p["geocent_time"]
    return np.array(
        [
            ifo.antenna_response(*args, p["psi"], "plus"),
            ifo.antenna_response(*args, p["psi"], "cross"),
            ifo.time_delay_from_geocenter(*args),
        ]
    )

def response(f, h, geo):
    """Combine the two polarizations to get the detector waveform."""
    return (geo[0] * h["plus"] + geo[1] * h["cross"]) * np.exp(-2j * np.pi * f * geo[2])

def read_psds(row, f, fmin, excluded):
    """Get PSD values and integration weights within the available frequency band."""
    data = []
    for det in row["pe_detectors"].split(","):
        item = row["_psds"][det]
        x, y = item["frequency"], item["psd"]
        mask = (f >= max(fmin, x[0])) & (f <= min(f[-1], x[-1])) & ~excluded
        psd = np.interp(f[mask], x, y)  # Linear interpolation inside the PSD band.
        df = f[1] - f[0]
        # Trapezoidal weights for 4*integral(df/PSD); keep gaps separate.
        neighbors = np.zeros(len(f))
        neighbors[1:] += mask[:-1]
        neighbors[:-1] += mask[1:]
        weights = 2 * df * neighbors[mask]
        data.append((det, bilby.gw.detector.get_empty_interferometer(det), mask, np.sqrt(weights / psd)))
    return data

def integrate_to(frequency, values, upper):
    """Integrate up to the given upper frequency using the trapezoidal rule."""
    keep = frequency <= upper
    x, y = frequency[keep], values[keep]
    if len(x) < 2:
        return 0.0
    if x[-1] < upper:
        x = np.append(x, upper)
        y = np.append(y, np.interp(upper, frequency, values))
    return float(np.trapezoid(y, x))

def selection_snr(row, f, h):
    """Calculate inspiral SNR from the waveform fraction and catalog SNR."""
    total, inspiral = 0.0, 0.0
    fmin, fhi, fc = float(row["fmin"]), FMAX, float(row["f_c"])
    for detector in row["pe_detectors"].split(","):
        item = row["_psds"][detector]
        x, y = item["frequency"], item["psd"]
        lo, hi = max(fmin, x[0]), min(fhi, x[-1])
        band = (f >= lo) & (f <= hi)
        band_f = f[band]
        band_psd = np.interp(band_f, x, y)
        geo = geometry(
            bilby.gw.detector.get_empty_interferometer(detector),
            {k: float(row[k]) for k in ("ra", "dec", "geocent_time", "psi")},
        )
        hd = response(band_f, {k: v[band] for k, v in h.items()}, geo)
        integrand = 4 * np.abs(hd) ** 2 / band_psd
        total += integrate_to(band_f, integrand, hi)
        inspiral += integrate_to(band_f, integrand, min(fc, hi))
    return float(row["snr_catalog"]) * np.sqrt(inspiral / total)

def calculate(row):
    """Calculate the three EFT bounds for one event, one coupling at a time."""
    p = {key: float(row[key]) for key in GR_PARAMETERS}
    fmin, fref = float(row["fmin"]), float(row["reference_frequency"])
    f = np.arange(round(FMAX / DF) + 1) * DF
    G, _ = phase_basis(f, p, fmin, fref, F=1.0, cycles=1.0)
    h = {key: np.zeros(len(f), complex) for key in ("plus", "cross")}
    hphi = {key: np.zeros(len(f), complex) for key in h}
    hb = [{key: np.zeros(len(f), complex) for key in h} for _ in BETAS]
    excluded = np.zeros(len(f), dtype=bool)
    for l, m in MODES:
        mode = gr_waveform(f, p, fmin, fref, [(l, m)])
        support = np.flatnonzero(abs(mode["plus"]) + abs(mode["cross"]))
        if len(support) and support[-1] < len(f) - 2:
            end = float(f[support[-1]])
            if CUTOFF_GUARD_HZ > 0:
                excluded |= abs(f - end) <= CUTOFF_GUARD_HZ
        for key in h:
            h[key] += mode[key]
            hphi[key] += 1j * m * mode[key]
            for j in range(3):
                hb[j][key] += 1j * G[m][j] * mode[key]
    # Exclude bins near sharp mode endings when computing all derivatives.
    psds = read_psds(row, f, fmin, excluded)
    rho_inspiral = selection_snr(row, f, h)
    # Vary masses, spins and inclination to get their waveform derivatives.
    shape = {}
    for name in ["mass_1", "mass_2", "chi_1z", "chi_2z", "iota"]:
        # Use four nearby points; mass derivatives are taken in log(mass).
        derivative = {key: np.zeros(len(f), complex) for key in h}
        for offset, coefficient in [(-2, 1), (-1, -8), (1, 8), (2, -1)]:
            shifted = dict(p)
            if name.startswith("mass"):
                shifted[name] *= np.exp(offset * STEP)
            else:
                shifted[name] += offset * STEP
            wave = gr_waveform(f, shifted, fmin, fref)
            for key in h:
                derivative[key] += coefficient * wave[key] / (12 * STEP)
        shape[name] = derivative
    blocks, signal = [], []
    for det, ifo, mask, weight in psds:
        geo = geometry(ifo, p)
        hd = response(f, h, geo)
        ds = [response(f, shape[name], geo) for name in ["mass_1", "mass_2", "chi_1z", "chi_2z"]]
        ds += [-hd, response(f, shape["iota"], geo)]
        # Vary sky position, polarization and time in the detector response.
        for name in ["ra", "dec", "psi", "geocent_time"]:
            step = 0.25 if name == "geocent_time" else STEP
            pp, pm = dict(p), dict(p)
            pp[name] += step
            pm[name] -= step
            dg = (geometry(ifo, pp) - geometry(ifo, pm)) / (pp[name] - pm[name])
            timing = dg[2] + (1.0 if name == "geocent_time" else 0.0)
            antenna = (dg[0] * h["plus"] + dg[1] * h["cross"]) * np.exp(-2j * np.pi * f * geo[2])
            ds.append(antenna - 2j * np.pi * f * timing * hd)
        ds += [response(f, hphi, geo)] + [response(f, b, geo) for b in hb]
        x = np.asarray(ds)[:, mask].T * weight[:, None]
        blocks.extend([x.real, x.imag])
        wh = hd[mask] * weight
        signal.extend([wh.real, wh.imag])
    # The first 11 columns are GR derivatives; the last three are EFT derivatives.
    A = np.vstack(blocks)
    rho = float(np.linalg.norm(np.concatenate(signal)))
    # Use SVD to marginalize over the eleven GR parameters.
    norms = np.linalg.norm(A[:, :11], axis=0)
    U, singular, _ = svd(A[:, :11] / norms, full_matrices=False, lapack_driver="gesvd")
    rank = int(np.sum(singular > RCOND * singular[0]))
    Q = U[:, :rank]
    out = {key: row[key] for key in ("event", "M_det", "eta", "f_c", "snr_catalog", "N_c", "far")}
    out.update(snr_point=rho, snr_inspiral=rho_inspiral)
    Msrc = (p["mass_1"] + p["mass_2"]) / (1 + float(row["redshift"]))
    out.update(M_src_point=Msrc, r_g_km=Msrc * RG_SUN_KM)
    # Convert each coupling error to a 90% length bound in km.
    for j, (beta, power) in enumerate(zip(BETAS, [4, 4, 6])):
        g = A[:, 11 + j]
        residual = g - Q @ (Q.T @ g)
        sigma = float(1 / np.linalg.norm(residual))
        out[beta + "_sigma"] = sigma
        out[beta + "_abs90"] = Z90 * sigma
        out[beta + "_ell90_km"] = out["r_g_km"] * (Z90 * sigma) ** (1 / power)
    return out


RESULT_FIELDS = [
    "event", "rank_beta1", "rank_beta2", "rank_beta3", "M_det", "M_src_point", "eta",
    "f_c", "snr_catalog", "snr_inspiral", "snr_point", "N_c", "far",
]
for beta in BETAS:
    RESULT_FIELDS += [beta + "_sigma", beta + "_abs90", beta + "_ell90_km"]


def write_results(path, rows):
    """Rank events by each length bound and save the results."""
    for beta in BETAS:
        ordered = sorted(rows, key=lambda row: float(row[beta + "_ell90_km"]))
        for rank, row in enumerate(ordered, 1):
            row["rank_" + beta] = rank
    rows.sort(key=lambda row: row["rank_beta2"])
    write_table(path, rows, RESULT_FIELDS)


def write_table(path, rows, fields):
    """Write a table with a header and tab-separated columns."""
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


# Read the selected events, calculate their bounds and save the results.
rows = read_events(EVENT_FILE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

events = []
for row in rows:
    if row["selected"] != "1":
        continue
    events.append(row)

results = []
for i, row in enumerate(events, 1):
    load_psds(row, PSD_ROOT)
    result = calculate(row)
    results.append(result)
    print(f"[{i}/{len(events)}] {row['event']}", flush=True)
path = OUTPUT_DIR + "fisher.txt"
write_results(path, results)
print("Saved:", path)