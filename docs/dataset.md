# Dataset datasheet

Description of the simulated dataset, following the structure of *Datasheets for
Datasets*. The numbers of a specific release (class balance, failed simulations)
are in the `report.md` that `scripts/dataset_report.py` writes inside each dataset
folder; this file describes what does not change between releases.

## Motivation

The dataset supports the study of in-service self-test of ECG analog front-ends: from
measurements the instrument can take on itself, decide whether the front-end still
meets its specifications, locate the faulty component and tell circuit faults from
electrode problems. It is entirely simulated; no patient data are involved.

## Composition

One folder per circuit (`integrated`, `reference`), each with:

| File | Content |
|---|---|
| `samples.parquet` | one row per simulated case |
| `waveforms.npy` | float32 array `[cases, 1000]`: response to the 1 mV calibration pulse, 1 s at 1 kHz, row-aligned with the table |
| `manifest.json` | full configuration, ngspice and code versions, git commit, counts |
| `report.md` | integrity checks and class balance of that release |

Each case is one circuit realisation: a healthy circuit drawn within manufacturing
tolerances, with at most one injected fault, connected to three electrodes drawn
from a gel or dry family; dry electrodes take the contact resistance measured on one
of six subjects. With the default configuration there are 5,000 healthy
cases and 200 cases for each fault condition (293 conditions for `integrated`, 307
for `reference`).

### Columns of `samples.parquet`

| Columns | Meaning |
|---|---|
| `sample_id`, `condition_index`, `replica` | identifiers; the random stream of a case is derived from `(seed, condition_index, replica)` |
| `sim_ok` | False if ngspice failed; the remaining columns are then empty |
| `condition`, `kind`, `target`, `level` | what was injected: fault type, component or electrode, magnitude |
| `is_faulty` | a fault was injected (percentage-severity view) |
| `compliant`, `violated`, `ok_<spec>` | level 1: the circuit meets every specification; which ones fail |
| `spec_<spec>` | continuous value of each specification (see docs/circuit.md) |
| `target` | level 2: component to locate |
| `origin` | level 3: `none`, `circuit` or `electrode` |
| `electrode_type`, `electrode_kind` | electrode family and type of the case |
| `p_<name>` | realised value of every component, op-amp, INA and electrode parameter |
| `dc_<node>` | C1: DC voltage at the output, the INA output and the RLD output [V] |
| `acd_mag_<f>`, `acd_ph_<f>` | C2: differential gain [V/V] and phase [deg] through the calibration source |
| `acc_mag_<f>` | C2: gain from the common-mode test source [V/V] |
| `zlo_mag_<f>` | C4: output per unit of lead-off test current [V/A] |

Features in the table and the waveforms are **noise-free**. Measurement noise, ADC
quantisation and clipping are applied when loading for an experiment
(`ecgfd.measurement.apply_measurement_model`), which also adds the CMRR estimates
(`cmrr_db_<f>`) and the pulse descriptors (`pulse_*`, C3).

## Generation process

`ecgfd --circuit <name> generate --out <folder>`. For each case, one ngspice run
takes the self-test measurements with the patient's electrodes, and a second run
measures the specifications with the test networks of IEC 60601-2-25. Generation is
deterministic for a given configuration and independent of the number of parallel
jobs. Models, values and sources are documented in [circuit.md](circuit.md).

## Labels

- **Functional** labels come from simulated specification values compared with the
  limits in the configuration. Changing only limits does not require new simulations
  (`ecgfd relabel`).
- Specifications are measured on a standard test network, so electrode faults leave
  the case compliant; they change the self-test features and are identified by `origin`.
- Many injected faults leave the circuit compliant. This is intended: it is what
  separates functional severity from percentage severity.

## Recommended use

- Split with `ecgfd.evaluation.replica_split` (every condition in train and test) or
  `magnitude_split` (whole parametric magnitudes held out). Never split after applying
  noise with different seeds to the same case.
- Classes are imbalanced by construction (few healthy and electrode cases against
  many circuit-fault cases). Report escape and false-reject rates, not only accuracy.
- Healthy cases with porous dry electrodes show a strongly reduced gain in service;
  see `report.md` and the open decision in [pendientes.md](pendientes.md).

## Limitations

- Simulation only: behavioural models of the amplifiers, ideal self-test sources,
  single faults, no ageing.
- Frequency-domain features are small-signal values.
- Dry electrodes follow six measured subjects per material; the spread between the
  electrodes of one case and the gel electrode parameters are assumed.
- The full list of open points is in [pendientes.md](pendientes.md).

## Distribution

Archived on Zenodo: <https://doi.org/10.5281/zenodo.23134950>. To be completed at publication:
version, licence and citation.
