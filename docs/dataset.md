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

One folder per circuit (`integrated`, `reference`), each a `spicefault` dataset:

| File | Content |
|---|---|
| `samples.parquet` | two rows per simulated case, one per operating condition |
| `waveforms.npy` | float32 array `[cases, 1000]`: response to the 1 mV calibration pulse, 1 s at 1 kHz. Only the `service` rows store it (`waveform_conditions` in the manifest); `spicefault.Dataset.waveforms` reads it aligned with the table, with NaN in the `bench` rows |
| `circuit.cir` | nominal netlist of the circuit |
| `metadata.json` | definition of the campaign: seed, faults, variations, conditions, analyses, measurements |
| `manifest.json` | fingerprints of the files, software versions, counts, the specification limits and the history of the labels; under `source`, the git commit of this repository, and under `user`, the study configuration |
| `report.md` | integrity checks and class balance of that release |

Each case is one circuit realisation: a healthy circuit drawn within manufacturing
tolerances, with at most one injected fault, connected to three electrodes drawn
from a gel or dry family. With the default configuration there are 5,000 healthy
cases and 200 cases for each fault (293 faults for `integrated`, 307 for `reference`).

A case is simulated under two operating conditions, and each gives a row: `service`
(patient's electrodes, self-test measurements) and `bench` (IEC 60601-2-25 test
networks, specifications). `ecgfd.dataset.load_cases` (`spicefault.Dataset.cases`)
returns one row per case, with the pulse responses.

### Columns of `samples.parquet`

| Columns | Meaning |
|---|---|
| `sample_id`, `fault_index`, `replica`, `seed_key` | identifiers; the random stream of a case is derived from `(seed, fault_index, replica)` and is the same under both conditions |
| `condition` | operating condition of the row: `service` or `bench` |
| `status`, `message`, `sim_ok`, `elapsed_s` | outcome of the simulation; if it failed, the measurements are empty |
| `fault_id`, `fault_type`, `fault_location`, `fault_magnitude`, `fault_severity` | what was injected (`healthy` if nothing): identifier, type, netlist elements, magnitude. The full record of each fault is in `metadata.json` |
| `component` | level 2: component to locate (designator, or electrode) |
| `origin` | level 3: `circuit` or `electrode`; empty for healthy cases |
| `compliant`, `violated`, `ok_spec_<spec>` | level 1: the circuit meets every specification; which ones fail (the same in both rows of a case). Written by `spicefault.Dataset.label` from the limits kept in the manifest |
| `spec_<spec>` | `bench` rows: continuous value of each specification (see docs/circuit.md) |
| `electrode_type`, `electrode_kind` | electrode family and type of the case |
| `p_<element>_<parameter>` | value drawn for every component, amplifier and electrode parameter, before the fault |
| `dc_<node>` | `service` rows, C1: DC voltage at the output and at the extended nodes [V] |
| `acd_mag_<f>`, `acd_ph_<f>` | `service` rows, C2: differential gain [V/V] and phase [deg] through the calibration source |
| `acc_mag_<f>` | `service` rows, C2: gain from the common-mode test source [V/V] |
| `zlo_mag_<f>` | `service` rows, C4: output per unit of lead-off test current [V/A] |

Features in the table and the waveforms are **noise-free**. Measurement noise, ADC
quantisation and clipping are applied when loading for an experiment
(`ecgfd.measurement.apply_measurement_model`), which also adds the CMRR estimates
(`cmrr_db_<f>`) and the pulse descriptors (`pulse_*`, C3).

## Generation process

`ecgfd --circuit <name> generate --out <folder>`, a `spicefault` fault campaign. For
each case, one ngspice run takes the self-test measurements with the patient's
electrodes, and a second run measures the specifications with the test networks of
IEC 60601-2-25. Generation is deterministic for a given configuration and independent
of the number of parallel jobs; `spicefault.Dataset(<folder>).reproduce(...)` simulates
stored samples again and compares them. Models, values and sources are documented in
[circuit.md](circuit.md).

## Labels

- **Functional** labels come from simulated specification values compared with the
  limits in the configuration. Changing only limits does not require new simulations
  (`ecgfd relabel`); the manifest keeps the history of the labels.
- Specifications are measured on a standard test network, so electrode faults leave
  the case compliant; they change the self-test features and are identified by `origin`.
- Many injected faults leave the circuit compliant. This is intended: it is what
  separates functional severity from percentage severity.

## Recommended use

- Split the cases with `spicefault.dataset.split_by_replica` (every fault in train and
  test) or `split_by_magnitude` (whole magnitudes held out). Never split after applying
  noise with different seeds to the same case.
- Classes are imbalanced by construction (few healthy and electrode cases against
  many circuit-fault cases). Report escape and false-reject rates, not only accuracy.
- Healthy cases with porous dry electrodes show a strongly reduced gain in service;
  see `report.md` and the open decision in [pendientes.md](pendientes.md).

## Limitations

- Simulation only: behavioural models of the amplifiers, ideal self-test sources,
  single faults, no ageing.
- Frequency-domain features are small-signal values.
- The spread of the electrode parameters around the published medians is assumed.
- The full list of open points is in [pendientes.md](pendientes.md).

## Distribution

To be completed at publication: version, DOI (Zenodo), licence and citation.
