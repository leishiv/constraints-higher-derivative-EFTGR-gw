#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tao Zhu (project additions).
"""Run the Bayesian analysis of GW230627_015337 with local strain and PSD files."""

import bilby
from gwpy.timeseries import TimeSeries
from eft_waveform import BETAS, waveform_eft

event = "GW230627_015337"
coupling = "beta2"  # Use "gr", "beta1", "beta2" or "beta3"; run one model at a time.

# Set the data folders and files here. Keep the final / in each folder path.
strain_dir = "./strain/"
psd_dir = "./psds/"
strain_files = {
    "H1": strain_dir + "H-H1_GWOSC_O4a_4KHZ_R1-1371865088-4096.hdf5",
    "L1": strain_dir + "L-L1_GWOSC_O4a_4KHZ_R1-1371865088-4096.hdf5",
}
psd_files = {
    "H1": psd_dir + "GW230627_015337_psd_H1.txt",
    "L1": psd_dir + "GW230627_015337_psd_L1.txt",
}
prior_file = "./code/GW230627_015337_eft.prior"
outdir = "./output/bayes/" + coupling
label = event + "_" + coupling + "_only"

# Data segment and analysis frequencies.
detectors = ["H1", "L1"]
trigger_time = 1371866035.8
start_time = trigger_time - 14
end_time = trigger_time + 2
duration = end_time - start_time
sampling_frequency = 4096
minimum_frequency = 20.0
maximum_frequency = 1024.0
reference_frequency = 20.0


# Read strain and PSD files for each detector.
ifo_list = bilby.gw.detector.InterferometerList([])
for detector in detectors:
    data = TimeSeries.read(
        strain_files[detector], format="hdf5.gwosc",
        start=start_time, end=end_time,
    )
    ifo = bilby.gw.detector.get_empty_interferometer(detector)
    ifo.strain_data.set_from_gwpy_timeseries(data)
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        psd_file=psd_files[detector]
    )
    ifo.minimum_frequency = minimum_frequency
    ifo.maximum_frequency = maximum_frequency
    ifo_list.append(ifo)

# Read the GR priors and set the coalescence-time prior.
priors = bilby.gw.prior.BBHPriorDict(filename=prior_file)
priors["geocent_time"] = bilby.core.prior.Uniform(
    trigger_time - 0.5, trigger_time + 0.5, name="geocent_time"
)
# Vary one coupling and fix the others to zero. beta3 uses the code convention.
for beta in BETAS:
    if beta == coupling:
        priors[beta] = bilby.core.prior.Uniform(-20.0, 20.0, name=beta)
    else:
        priors[beta] = bilby.core.prior.DeltaFunction(0.0, name=beta)

# Generate the waveform with the EFT phase correction.
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    start_time=start_time,
    frequency_domain_source_model=waveform_eft,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=dict(
        waveform_approximant="SEOBNRv5HM_ROM",
        minimum_frequency=minimum_frequency,
        reference_frequency=reference_frequency,
        catch_waveform_errors=True,
        f_window_div_f_peak=1.0,
        num_cycles=1.0,
    ),
)

# Set the likelihood and marginalize over time and distance.
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list, waveform_generator, priors=priors,
    time_marginalization=True,
    phase_marginalization=False,
    distance_marginalization=True,
)

# Run the sampler and save the result.
result = bilby.run_sampler(
    likelihood, priors, sampler="nessai",
    outdir=outdir, label=label,
    nlive=1000, npool=64, seed=42,
    analytic_priors=True, stopping=0.1, resume=False,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
)
print("log evidence:", result.log_evidence, "+/-", result.log_evidence_err)
