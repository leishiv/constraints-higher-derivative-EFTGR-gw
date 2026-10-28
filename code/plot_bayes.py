#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tao Zhu (project additions).
"""Summarize four Bayesian results and draw Figures 1 and 3."""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from eft_waveform import BETAS, phase_basis

# Set the input files and output folder here. Keep the final / in folder paths.
sample_dir = "./data/posterior_samples/"
event_file = "./data/events.txt"
output_dir = "./output/"
GWID = "GW230627_015337"
LABEL = {b: rf"$\beta_{i}$ ({pn}PN)" for b, i, pn in zip(BETAS, (1, 2, 3), (6, 5, 8))}
RG_SUN_KM = 1.476625038050125
os.makedirs(output_dir, exist_ok=True)

# Read GR and EFT results. New runs may use a subfolder for each model.
posterior = {}
for tag in ["gr"] + BETAS:
    name = f"{GWID}_{tag}_only_result.json"
    path = sample_dir + name
    if not os.path.isfile(path):
        path = sample_dir + tag + "/" + name
    with open(path) as stream:
        data = json.load(stream)
    result = dict(
        lnZ=float(data["log_evidence"]), lnZ_error=float(data["log_evidence_err"])
    )
    if tag != "gr":

        columns = {}
        for key in (tag, "mass_1_source", "mass_2_source"):
            values = data["posterior"]["content"][key]
            if isinstance(values, dict):
                values = [values[str(i)] for i in range(len(values))]
            columns[key] = np.asarray(values, float)
        # Convert beta3 to the paper convention; keep the paired source masses.
        result["beta"] = columns[tag] / (3.0 if tag == "beta3" else 1.0)
        result["mass"] = columns["mass_1_source"] + columns["mass_2_source"]
    posterior[tag] = result

# Calculate the coupling intervals, length bounds and Bayes factors.
rows = []
for beta, power in zip(BETAS, (4, 4, 6)):
    result = posterior[beta]
    b, mass = result["beta"], result["mass"]
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(mass)):
        raise ValueError("Nonfinite posterior samples: " + beta)
    lo, med, hi = np.percentile(b, [5, 50, 95])
    length = RG_SUN_KM * mass * np.abs(b) ** (1 / power)
    rows.append(
        dict(
            coupling=beta,
            median=med,
            lower90=lo,
            upper90=hi,
            abs90=np.percentile(abs(b), 90),
            ell90_km=np.percentile(length, 90),
            lnB=result["lnZ"] - posterior["gr"]["lnZ"],
            lnB_error=np.hypot(result["lnZ_error"], posterior["gr"]["lnZ_error"]),
        )
    )
with open(output_dir + "bayes.txt", "w", newline="") as stream:
    writer = csv.DictWriter(
        stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(
        row["coupling"],
        f"median={row['median']:+.4f}, ell90={row['ell90_km']:.2f} km, lnB={row['lnB']:+.3f}",
    )

# Plot Figure 3: coupling posteriors.
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
for ax, k in zip(axes, BETAS):
    b = posterior[k]["beta"]
    lo, med, hi = np.percentile(b, [5, 50, 95])
    ax.hist(b, bins=60, density=True, histtype="step", color="C0", linewidth=1.5)
    ax.axvspan(lo, hi, color="C0", alpha=0.15, zorder=0)
    ax.axvline(0.0, color="0.25", linestyle="--", linewidth=1.1)
    ax.axvline(med, color="C0", linewidth=1.5)
    err_lo, err_hi = med - lo, hi - med
    legend_text = (
        rf"$\beta_{k[-1]} = {med:+.3f}^{{+{err_hi:.3f}}}" rf"_{{-{err_lo:.3f}}}$"
    )
    ax.legend(
        handles=[Line2D([], [], color="C0", label=legend_text)],
        loc="upper right",
        frameon=False,
        fontsize=10,
        handlelength=0,
        handletextpad=0,
        borderaxespad=0.3,
    )
    ax.set_xlabel(LABEL[k], fontsize=10)
    ax.set_yticks([])
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", top=True, right=True, labelsize=10)
fig.tight_layout(w_pad=1.0)
fig.savefig(output_dir + "fig_post.pdf", bbox_inches="tight")
plt.close(fig)

# Plot Figure 1: phase corrections at the event median values.
with open(event_file, newline="") as stream:
    row = next(
        r for r in csv.DictReader(stream, delimiter="\t") if r["event"] == GWID
    )
p = {k: float(row[k]) for k in ("mass_1", "mass_2", "chi_1z", "chi_2z")}
frequency = np.arange(round(4095.75 / 0.125) + 1) * 0.125
phases, peak = phase_basis(
    frequency,
    p,
    float(row["fmin"]),
    float(row["reference_frequency"]),
    F=1.0,
    cycles=1.0,
)
keep = (frequency >= 20) & (frequency <= 1024)
labels = (r"$\beta_1$ (6PN)", r"$\beta_2$ (5PN + 6PN)", r"$\beta_3$ (8PN)")
fig, ax = plt.subplots(figsize=(6.0, 4.0))
for i, label in enumerate(labels):
    # The phase derivative is already in the paper convention, including beta3.
    ax.plot(frequency[keep], np.abs(phases[2][i, keep]), label=label)
ax.axvline(peak, ls=":", c="k", label=r"$f_{22}^{\rm peak}$")
ax.set(
    xscale="log",
    yscale="log",
    xlabel="frequency [Hz]",
    ylabel=r"$|Q_{22}|$ [rad]",
)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir + "fig_dephasing.pdf")
plt.close(fig)
