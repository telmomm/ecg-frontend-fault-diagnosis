# ecg-frontend-fault-diagnosis

Specification-aware fault diagnosis of ECG analog front-ends, based on simulation:
ngspice Monte Carlo with fault injection, and machine-learning models that use only
measurements the instrument could take on itself to answer three questions in
service. Does the front-end still meet its specifications? If not, which component
is the cause? Is the problem in the circuit or in the electrodes?

- Research plan (Spanish): [docs/linea_diagnostico_fallos_frontend_ecg.md](docs/linea_diagnostico_fallos_frontend_ecg.md)
- Literature review (Spanish): [docs/SOTA/](docs/SOTA/sota_diagnostico_fallos_frontend_ecg.md)
- Circuits, specifications and design record: [docs/circuit.md](docs/circuit.md)
- Open points and values still to be confirmed (Spanish): [docs/pendientes.md](docs/pendientes.md)

## Status

| Phase of the plan | State |
|---|---|
| 2. Circuits and specifications | Closed: INA333-based main circuit and discrete reference circuit, specifications from IEC 60601-2-25, both pass E1. Limits checked against the text of the standard; the open design decision on porous dry electrodes is in docs/pendientes.md |
| 3. Simulation pipeline | Closed: netlists, fault injection, Monte Carlo, specifications, features C1–C4, parallel and resumable generation, relabelling, tests |
| 4. Dataset | Closed: `data/v2` generated (63,600 + 66,400 cases, no failed simulation), checked with `make report`, datasheet in docs/dataset.md |
| 5. Testability (E2, E9) | Closed (`make testability`); results summarised in the plan |
| 6. Models (E3–E6) | Closed (`make models`, `make models-sensitivity`); H1, H2 and H4 assessed, results summarised in the plan |
| 7. Robustness (E7, E8) | Closed (`make shift-datasets`, `make robustness`); results summarised in the plan |

## Setup

Requires Python ≥ 3.10 and [ngspice](https://ngspice.sourceforge.io/) on the `PATH`
(developed with ngspice 44; `brew install ngspice` or `apt install ngspice`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # add ",dl" for the PyTorch CNN
pytest                         # about 5 s; ngspice tests are skipped if it is missing
```

## Usage

There are two circuits: `integrated` (main, the default, built around the INA333)
and `reference` (discrete three-op-amp amplifier). Pass `--circuit reference` to
work on the second one. Their schematics are in [docs/figures/](docs/figures/).

```bash
ecgfd nominal                  # self-test features and specifications of the nominal circuit
ecgfd netlist                  # print the ngspice deck
ecgfd faults                   # size of the fault catalogue

# Smoke datasets: about 1,200 simulations and 2.5 minutes each
make smoke

# Full datasets: 63,600 + 66,400 simulations, about 2.5 hours on 8 cores.
# An interrupted run resumes from the last complete chunk when launched again.
make dataset

# Integrity checks and class balance; writes report.md inside each dataset folder
make report

# After changing only specification limits: new labels, no new simulations
ecgfd relabel --data data/v2/integrated

python experiments/e1_nominal_validation.py                      # specs: nominal and healthy yield
python experiments/e2_ambiguity_groups.py --data data/v2/integrated   # testability
python experiments/e9_architecture.py --data-dir data/v2             # both circuits compared
python experiments/e3_spec_prediction.py  --data data/v2/integrated   # specs from measurements
python experiments/e4_severity.py         --data data/v2/integrated   # functional vs percentage
python experiments/e5_localisation.py     --data data/v2/integrated   # which component (--cnn)
python experiments/e6_origin.py           --data data/v2/integrated   # circuit vs electrode
```

Experiments write to `results/<experiment>/<circuit>/` (runs on the smoke datasets go
to `results/smoke/` instead). The scripts that read a
dataset accept `--electrode-kinds` to keep only some electrode types (results then go
to `<circuit>-<tag>`), e.g. to study the circuits without porous dry electrodes. Results on the smoke datasets
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
| `scripts/dataset_report.py` | Integrity checks and class balance of a generated dataset |
| `scripts/draw_schematics.py` | Draws both schematics into `docs/figures/` |
| `scripts/validate_ina_model.py` | Compares the behavioural INA with the TI INA333 macromodel (fetched by `scripts/fetch_vendor_models.py`) |
| `tests/` | Unit and end-to-end tests, run on both circuits |

## How a dataset is built

1. Each simulation gets its own random stream from `(seed, condition, replica)`, so the
   result does not depend on the number of parallel jobs.
2. A healthy circuit and its electrodes (gel or dry) are drawn, then one fault is injected.
3. A first ngspice run takes the self-test measurements in service, with the
   patient's electrodes: operating point, three frequency responses and the response
   to the 1 mV calibration pulse.
4. A second run measures the specifications with the IEC 60601-2-25 test networks. Comparing
   them with the limits gives the functional label; the injected fault gives the
   localisation and origin labels.
5. The noise-free results go to `samples.parquet` and `waveforms.npy`;
   `manifest.json` keeps the configuration and the software versions.
6. Measurement noise and ADC quantisation are applied when the features are loaded for
   an experiment, so they are study parameters.

## Design choice: ngspice driven directly

ngspice is launched from Python without PySpice (netlist text in, binary raw file
out, about 100 lines in `spice.py`): fault injection needs free editing of the
netlist, and it avoids a dependency that lags behind ngspice releases.

## Licence

Code under the MIT licence (see `LICENSE`). The dataset will be published separately.
