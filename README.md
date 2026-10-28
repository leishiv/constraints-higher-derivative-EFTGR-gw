# Gravitational-wave tests of higher-derivative gravity

This project uses gravitational-wave data to constrain cubic and quartic curvature corrections to general relativity. The corresponding 5PN, 6PN and 8PN phase corrections are added to the SEOBNRv5HM_ROM waveform using the flexible theory-independent (FTI) method. Fisher calculations rank 92 selected binary black hole events by their expected bounds on three characteristic length scales. A Bayesian analysis of GW230627_015337, the event with the tightest Fisher bounds, estimates the couplings and compares each model with general relativity.

## Paper

This repository accompanies the paper:

Tao Zhu, [Improved constraints on higher-derivative effective field theory extensions of general relativity with GW230627_015337](https://arxiv.org/abs/2610.36728), arXiv:2610.36728 [gr-qc] (2026).

## Files

- `code/fisher.py`: calculate Fisher bounds and rank the 92 events.
- `code/run_bayes.py`: run one Bayesian model with `waveform_eft`.
- `code/plot_fisher.py`: plot the Fisher results.
- `code/plot_bayes.py`: summarize the Bayesian results and plot the posteriors and phase corrections.
- `code/eft_waveform.py`: GR waveform and EFT phase correction.
- `data/events.txt`: GR medians, selection flags and data sources for 273 events; 92 have `selected = 1`.
- `data/posterior_samples/`: four original Bayesian result JSONs.

The GR prior is in `code/GW230627_015337_eft.prior`. Reference results and figures are in `results/` and `figures/`.

## Setup

```bash
conda env create -f environment.yml
conda activate cubic-eft
export LAL_DATA_PATH=/path/to/ROM_data
```

Download `SEOBNRv5HMROM_v1.0.hdf5` from the [ROM data release](https://zenodo.org/records/14999310) and put it in the directory set by `LAL_DATA_PATH`. The file is about 2.2 GB. Figure 2 also needs a LaTeX installation.

## Data

Strain, PSD and PE files are not included. The scripts do not download them.

### PSD files

Each TXT file has two columns: frequency in Hz and one-sided PSD in strain squared per Hz. Use PSD, not ASD. Name files as `<event>_psd_<detector>.txt`, for example `GW230627_015337_psd_H1.txt`. The 92 events need 229 files. Use only the detectors listed in `pe_detectors`.

`data/events.txt` gives the official PE download URL (`pe_url`), filename (`pe_file`) and analysis group (`pe_group`). Download the recorded version. The PSD array is at `<pe_group>/psds/<detector>` in the HDF5 file. Different analysis groups can have different PSDs. Keep the original grid and values; do not extrapolate.

This example exports the selected PSDs. Change `pe_root` and run from the project root:

```python
import csv
import os
import h5py
import numpy as np

pe_root = "/path/to/PEDataRelease/"
psd_dir = "./psds/"
os.makedirs(psd_dir, exist_ok=True)
with open("./data/events.txt", newline="") as stream:
    events = list(csv.DictReader(stream, delimiter="\t"))
for event in events:
    if event["selected"] != "1":
        continue
    with h5py.File(pe_root + event["pe_file"], "r") as pe:
        for detector in event["pe_detectors"].split(","):
            array = pe[event["pe_group"] + "/psds/" + detector][:]
            name = event["event"] + "_psd_" + detector + ".txt"
            np.savetxt(psd_dir + name, array, fmt="%.18e")
```

After export, the calculations need only the PSD TXT files. They do not read PE files. PSDs estimated from strain are not an exact replacement for these PE PSDs.

### Strain files

For the Bayesian run, obtain these 4096-Hz files from [GWOSC](https://gwosc.org/) or its [API](https://gwosc.org/apidocs/):

```text
H-H1_GWOSC_O4a_4KHZ_R1-1371865088-4096.hdf5
L-L1_GWOSC_O4a_4KHZ_R1-1371865088-4096.hdf5
```

The script reads a 16-second segment around GW230627_015337.

## Run

Set the paths near the top of each script. Keep the final `/` in folder paths. Run from the project root, or use absolute paths.

- `fisher.py`: set `EVENT_FILE`, `PSD_ROOT` and `OUTPUT_DIR`. Results go to `OUTPUT_DIR + "fisher.txt"`, sorted by the beta2 length bound. All three ranks and length bounds in km are saved.
- `run_bayes.py`: set `strain_dir`, `psd_dir`, `prior_file` and `outdir`. Run separately with `coupling` set to `gr`, `beta1`, `beta2` and `beta3`. Sampler settings are in `bilby.run_sampler()`. Set `npool` to match your server resources.
- `plot_fisher.py`: set `table_file` and `output_dir`. The default reads `./results/fisher.txt`; use `./output/fisher.txt` for new results.
- `plot_bayes.py`: set `sample_dir`, `event_file` and `output_dir`. The default reads the supplied JSONs. Set `sample_dir = "./output/bayes/"` to read four new runs.

```bash
python code/fisher.py
python code/run_bayes.py
python code/plot_fisher.py
python code/plot_bayes.py
```

Run only the command you need. The plotting scripts need no strain or PSD files. They save `fig_mass.pdf`, or `fig_post.pdf`, `fig_dephasing.pdf` and `bayes.txt`, in their chosen output folders.

## Notes

- The scripts execute directly when opened by Python. Do not import them as modules. Parallel Bayesian runs use Linux `fork`; `spawn` and `forkserver` need an entry-point guard.
- `waveform_eft` and the saved JSONs use `beta3_code = 3 * beta3_reported`. The Bayesian summary and plots divide the samples by three once.
- The model includes EFT phase terms only. EFT amplitude and spin terms are not considered.

## Acknowledgments

Codex and WorkBuddy were used to assist with code writing, numerical calculations and other tasks during this project. The authors are responsible for the methods, results and final code.

## License

Code: GPL v3 or later. Project data and figures: CC BY 4.0.
Third-party material keeps its original terms. See `LICENSE` for all notices.
