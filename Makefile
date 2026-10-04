PY ?= .venv/bin/python
DATA ?= data/v2
CIRCUITS = integrated reference

.PHONY: setup test lint smoke dataset report e1 testability models models-quick models-sensitivity shift-datasets robustness all-experiments paper

setup:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

smoke:
	for c in $(CIRCUITS); do \
		$(PY) -m ecgfd.cli --config configs/smoke.yaml --circuit $$c generate --out data/smoke/$$c; \
	done

dataset:
	for c in $(CIRCUITS); do \
		$(PY) -m ecgfd.cli --circuit $$c generate --out $(DATA)/$$c; \
	done

# integrity checks and class balance of the datasets under $(DATA)
report:
	for c in $(CIRCUITS); do $(PY) scripts/dataset_report.py --data $(DATA)/$$c; done

e1:
	for c in $(CIRCUITS); do $(PY) experiments/e1_nominal_validation.py --circuit $$c; done

# Phase 5: E2 on both circuits and E9, with all electrodes and without porous dry ones
SOLID = ag_agcl_gel stainless_steel silver platinum
testability:
	for c in $(CIRCUITS); do \
		$(PY) experiments/e2_ambiguity_groups.py --data $(DATA)/$$c; \
		$(PY) experiments/e2_ambiguity_groups.py --data $(DATA)/$$c --electrode-kinds $(SOLID) --tag gel-solid; \
	done
	$(PY) experiments/e9_architecture.py --data-dir $(DATA)
	$(PY) experiments/e9_architecture.py --data-dir $(DATA) --electrode-kinds $(SOLID) --tag gel-solid

# Phase 6: E3-E6 on both circuits with every electrode type, then E9 with the E5 scores
MODELS = e3_spec_prediction e4_severity e6_origin
models:
	for c in $(CIRCUITS); do \
		for e in $(MODELS); do $(PY) experiments/$$e.py --data $(DATA)/$$c; done; \
		$(PY) experiments/e5_localisation.py --data $(DATA)/$$c --cnn; \
	done
	$(PY) experiments/e9_architecture.py --data-dir $(DATA)

# E3, E4 and E6 only (minutes); E5, the slow one, is left as it is
models-quick:
	for c in $(CIRCUITS); do \
		for e in $(MODELS); do $(PY) experiments/$$e.py --data $(DATA)/$$c; done; \
	done

# Sensitivity of phase 6 to the electrodes: type given to the models, and no porous dry ones
models-sensitivity:
	for c in $(CIRCUITS); do \
		for e in $(MODELS) e5_localisation; do \
			$(PY) experiments/$$e.py --data $(DATA)/$$c --known-electrode; \
			$(PY) experiments/$$e.py --data $(DATA)/$$c --electrode-kinds $(SOLID) --tag gel-solid; \
		done; \
	done

# Phase 7. Small test-only datasets generated with other tolerance settings (about an hour)
SHIFTS = truncnorm tolerance
shift-datasets:
	for c in $(CIRCUITS); do \
		for s in $(SHIFTS); do \
			$(PY) -m ecgfd.cli --config configs/shift_$$s.yaml --circuit $$c generate --out data/shift/$$s/$$c; \
		done; \
	done

# Phase 7: E7 (unseen magnitudes, noise and quantisation, other tolerances) and E8 (minimal tests)
robustness:
	for c in $(CIRCUITS); do \
		$(PY) experiments/e7_robustness.py --data $(DATA)/$$c \
			--shift-data data/shift/truncnorm/$$c data/shift/tolerance/$$c; \
		$(PY) experiments/e8_minimal_tests.py --data $(DATA)/$$c; \
		$(PY) experiments/e8_minimal_tests.py --data $(DATA)/$$c --strict; \
	done

# Every experiment of phases 5 to 7 on existing datasets (several hours)
all-experiments: testability models models-sensitivity robustness

# Figures of the manuscript from results/, then the PDF (needs a LaTeX installation)
paper:
	$(PY) scripts/draw_schematics.py
	$(PY) scripts/paper_figures.py
	cd manuscript && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex supplement.tex
