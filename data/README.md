# Simulated ECG Analog Front-End Fault-Diagnosis Data

This deposit contains simulation datasets for studying specification-aware
self-test and fault diagnosis of ECG analog front-ends. The data support three
tasks: determining whether a circuit meets its specifications, locating an
injected fault, and distinguishing circuit faults from electrode-related
problems.

All records are generated with ngspice. They are not recordings from patients,
and contain no personal or clinical data. The simulations include component
tolerances, injected component or electrode faults, and models of gel and dry
electrodes. The two circuit architectures are:

- `integrated`: the main single-supply front-end based on an INA333 and a
	discrete network.
- `reference`: a discrete three-operational-amplifier instrumentation
	amplifier used as a reference architecture.

## Contents

Each dataset release is organised by circuit. A circuit directory contains:

- `samples.parquet`: one row per simulated case, with injected-condition
	identifiers, fault and compliance labels, simulated specifications,
	measurement features, and realised component/electrode parameters.
- `waveforms.npy`: `float32` calibration-pulse waveforms, aligned row-for-row
	with `samples.parquet`.
- `manifest.json`: dataset configuration, random seed, software and ngspice
	versions, source-code commit, and generation counts.
- `report.md`: integrity checks and class-balance summary for that circuit.

The waveform array has shape `[number of cases, 1000]`; each row represents a
1-second response sampled at 1 kHz. The stored measurements are noise-free.
Noise, ADC quantisation, and clipping are applied by the analysis code when
features are loaded for an experiment.

## Dataset releases

- `v1/`: the first full release, containing `integrated` and `reference`
	datasets. Retained to make analyses based on the initial model reproducible.
- `v2/`: the updated full release used for the main study, with both circuit
	architectures. It updates the dry-electrode model to include contact
	resistance values associated with six measured subjects. The `manifest.json`
	files record the exact configuration and sample counts for each circuit.
- `shift/tolerance/`: robustness-test data generated with a different random
	seed and wider component tolerances (2% for resistors and 10% for
	capacitors), to evaluate models trained on the default-tolerance data.
- `shift/truncnorm/`: robustness-test data generated with a different random
	seed and truncated-normal rather than uniform component-tolerance sampling.
- `smoke/`: small pipeline-check datasets for both circuits. These are useful
	for testing data loading and analysis workflows, but are not sized for
	scientific performance claims.

Each group contains both circuit directories unless stated otherwise. Use the
manifest in each directory as the authoritative record of that dataset's
settings and counts.

## Loading the data

From the root of the software repository, the project loader reads a circuit
directory and keeps the table and waveforms aligned:

```python
from ecgfd.dataset import load_dataset

dataset = load_dataset("data/v2/integrated")
```

The project source code and experiment scripts are available in the
[associated software repository](https://github.com/telmomm/ecg-frontend-fault-diagnosis).
The manifest records the source commit used to generate each dataset. Analysis
requires Python and the dependencies documented by that repository; ngspice is
required to regenerate simulations, not to load the published files. If this
data folder is downloaded separately from the source repository, adjust the
path passed to `load_dataset` to match its extracted location.

## Scope and limitations

These data represent circuit simulations, not measurements from a physical ECG
device. They use behavioural amplifier models, idealised self-test sources, and
single injected faults; ageing and multiple simultaneous faults are not
modelled. Dry-electrode variability uses measurements from six subjects and
should not be interpreted as representative of all users or electrodes.
Specification labels are calculated from simulated values using the limits
stored in each dataset's manifest.

## Citation

Please cite this Zenodo record and the associated software repository when
using the data. Add the final Zenodo citation and DOI here after publication.

## License

Recommended dataset license: **Creative Commons Attribution 4.0 International
(CC BY 4.0)**. Select this license in the Zenodo record and retain the required
attribution to the source used for the dry-electrode parameter values. The
software repository's MIT license applies to its code, not automatically to
these data.
