# ecg-frontend-fault-diagnosis

Specification-aware fault diagnosis of ECG analog front-ends, based on simulation:
ngspice Monte Carlo with fault injection, and machine-learning models that use only
measurements the instrument could take on itself to answer three questions in
service. Does the front-end still meet its specifications? If not, which component
is the cause? Is the problem in the circuit or in the electrodes?

- Research plan (Spanish): [docs/linea_diagnostico_fallos_frontend_ecg.md](docs/linea_diagnostico_fallos_frontend_ecg.md)
- Literature review (Spanish): [docs/SOTA/](docs/SOTA/sota_diagnostico_fallos_frontend_ecg.md)
- Circuits, specifications and open decisions: [docs/circuit.md](docs/circuit.md)

## Status

| Phase of the plan | State |
|---|---|
| 2. Circuits and specifications | Both circuits simulate and pass E1. Specification limits are placeholders until checked against the IEC standards |
| 3. Simulation pipeline | Done: netlists, fault injection, Monte Carlo, specifications, features C1–C4, parallel generation, tests |
| 4. Dataset | Not generated. Only the smoke datasets have been run |
| 5. Testability (E2, E9) | E2 first pass. E9 not written (a class-separability metric exists) |
| 6. Models (E3–E6) | Untuned baselines that run end to end |
| 7. Robustness (E7, E8) | Not written (the split by unseen magnitude exists) |

## Setup

Requires Python ≥ 3.10 and [ngspice](https://ngspice.sourceforge.io/) on the `PATH`
(developed with ngspice 44; `brew install ngspice` or `apt install ngspice`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # add ",dl" for the PyTorch CNN
pytest                         # about 5 s; ngspice tests are skipped if it is missing
```

## Usage

There are two circuits: `integrated` (main, the default) and `reference` (discrete
three-op-amp amplifier). Pass `--circuit reference` to work on the second one.

```bash
ecgfd nominal                  # self-test features and specifications of the nominal circuit
ecgfd netlist                  # print the ngspice deck
ecgfd faults                   # size of the fault catalogue

# Smoke datasets: about 1,200 simulations and 1.5 minutes each
make smoke

# Full datasets: 63,600 + 66,400 simulations, roughly 2.5 hours on 8 cores
make dataset

python experiments/e1_nominal_validation.py                      # specs: nominal and healthy yield
python experiments/e2_ambiguity_groups.py --data data/v1/integrated   # testability
python experiments/e3_spec_prediction.py  --data data/v1/integrated   # specs from measurements
python experiments/e4_severity.py         --data data/v1/integrated   # functional vs percentage
python experiments/e5_localisation.py     --data data/v1/integrated   # which component (--cnn)
python experiments/e6_origin.py           --data data/v1/integrated   # circuit vs electrode
```

Experiments write to `results/<experiment>/<circuit>/`. Results on the smoke datasets
only show that the code runs: with four samples per condition they say nothing about
diagnosability.

## Layout

| Path | Content |
|---|---|
| `configs/` | Study configuration: circuits, tolerances, electrodes, specification limits, fault levels, measurements, dataset size |
| `src/ecgfd/circuit.py` | Topology of both circuits, nominal values, netlist generation |
| `src/ecgfd/spice.py` | ngspice batch runner and raw-file reader |
| `src/ecgfd/sampling.py` | Monte Carlo sampling of healthy circuits and electrodes |
| `src/ecgfd/faults.py` | Fault catalogue and injection |
| `src/ecgfd/simulate.py` | Self-test measurements: DC, AC, lead-off current, calibration pulse |
| `src/ecgfd/specs.py` | Specifications of each case and compliance labels |
| `src/ecgfd/measurement.py` | ADC noise, quantisation and clipping |
| `src/ecgfd/features.py` | Feature sets C1–C4 |
| `src/ecgfd/dataset.py` | Parallel dataset generation and loading |
| `src/ecgfd/ambiguity.py` | Sensitivities, fault dictionary, ambiguity groups |
| `src/ecgfd/evaluation.py` | Escape and false-reject rates, class separability, leakage-free splits |
| `src/ecgfd/models/` | Reference classifiers and the 1D CNN |
| `experiments/` | One script per experiment of the plan |
| `tests/` | Unit and end-to-end tests, run on both circuits |

## How a dataset is built

1. Each simulation gets its own random stream from `(seed, condition, replica)`, so the
   result does not depend on the number of parallel jobs.
2. A healthy circuit and its electrodes (gel or dry) are drawn, then one fault is injected.
3. A first ngspice run takes the self-test measurements in service, with the
   patient's electrodes: operating point, three frequency responses and the response
   to the 1 mV calibration pulse.
4. A second run measures the specifications on a standard test network. Comparing
   them with the limits gives the functional label; the injected fault gives the
   localisation and origin labels.
5. The noise-free results go to `samples.parquet` and `waveforms.npy`;
   `manifest.json` keeps the configuration and the software versions.
6. Measurement noise and ADC quantisation are applied when the features are loaded for
   an experiment, so they are study parameters.

## Design choice: no PySpice

The plan suggests PySpice. The repository drives ngspice directly instead (netlist
text in, binary raw file out, about 100 lines in `spice.py`): fault injection needs
free editing of the netlist, and it avoids a dependency that lags behind ngspice releases.

## Licence

Code under the MIT licence (see `LICENSE`). The dataset will be published separately.
