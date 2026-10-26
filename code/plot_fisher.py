#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tao Zhu (project additions).
"""Plot Figure 2 from the Fisher result table."""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

# Set the input table and output folder here.
table_file = "./results/fisher.txt"
output_dir = "./figures/"  # Keep the final /.
output_file = output_dir + "fig_mass.pdf"
GWID = "GW230627_015337"
KEYS = ("beta1", "beta2", "beta3")

# Read the table and sort by event name.
with open(table_file, newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
if not rows:
    raise ValueError("No Fisher results in " + table_file)
rows = sorted(rows, key=lambda row: row["event"])
os.makedirs(output_dir, exist_ok=True)

# Set the fonts used in the paper.
style = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.latex.preamble": r"\usepackage{underscore}",
}
plt.rcParams.update(style)

# Draw the three panels and the shared SNR colorbar.
norm = matplotlib.colors.PowerNorm(
    gamma=0.30,
    vmin=6.0,
    vmax=max(30.0, max(float(r["snr_inspiral"]) for r in rows)),
)
fig, axes = plt.subplots(1, 3, figsize=(7.06, 2.65), sharex=True, sharey=True)
# Reserve a separate column for the colorbar before placing any axes.
fig.subplots_adjust(left=0.075, right=0.865, bottom=0.24, top=0.96, wspace=0.13)
special = next((r for r in rows if r["event"] == GWID), None)
mass = np.array([float(r["M_det"]) for r in rows])
snr = np.array([float(r["snr_inspiral"]) for r in rows])
for i, (ax, key) in enumerate(zip(axes, KEYS)):
    ell = np.array([float(r[key + "_ell90_km"]) for r in rows])
    sc = ax.scatter(
        mass,
        ell,
        c=snr,
        cmap="turbo",
        norm=norm,
        s=28,
        edgecolors="white",
        linewidths=0.3,
        zorder=2,
    )
    if special is not None:
        sx, sy = float(special["M_det"]), float(special[key + "_ell90_km"])
        ax.scatter(
            [sx],
            [sy],
            marker="*",
            s=120,
            c="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )
        ax.annotate(
            GWID.replace("_", r"\_"),
            xy=(sx, sy),
            xytext=(0.96, 0.15),
            textcoords="axes fraction",
            fontsize=9,
            ha="right",
            va="center",
            bbox=dict(
                boxstyle="round,pad=.15",
                facecolor="white",
                alpha=0.92,
                edgecolor="0.7",
                linewidth=0.4,
            ),
            arrowprops=dict(arrowstyle="->", lw=0.65, color="0.25"),
            zorder=5,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([10, 20, 50, 100], labels=["10", "20", "50", "100"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(9.5, 105)
    ax.yaxis.set_major_locator(FixedLocator([10, 20, 30, 50, 100]))
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.text(
        0.04,
        0.96,
        r"$\beta_%d$" % (i + 1),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
            boxstyle="square,pad=.12",
        ),
        zorder=5,
    )
    ax.set_xlabel(r"$M_{\rm det}\,[M_\odot]$", fontsize=9, labelpad=3)
    ax.grid(alpha=0.22, which="major")
    ax.tick_params(direction="in", top=True, right=True, labelsize=9, pad=3)
axes[0].set_ylabel(r"$\ell_{90}$ [km]", fontsize=9, labelpad=3)
color_ax = fig.add_axes([0.902, 0.24, 0.017, 0.72])
cbar = fig.colorbar(sc, cax=color_ax, ticks=[6, 10, 20, 30])
cbar.ax.set_yticklabels(["6", "10", "20", "30"])
cbar.set_label(r"$\rho_{\rm insp}$", fontsize=9, labelpad=4)
cbar.ax.tick_params(labelsize=9, pad=3)
fig.savefig(output_file)
plt.close(fig)
print("Saved:", output_file)
