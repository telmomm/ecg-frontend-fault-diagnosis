PY ?= .venv/bin/python
DATA ?= data/v1
CIRCUITS = integrated reference

.PHONY: setup test lint smoke dataset report e1 experiments

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

# E2-E6 on the datasets under $(DATA)
experiments:
	for c in $(CIRCUITS); do \
		for e in e2_ambiguity_groups e3_spec_prediction e4_severity e5_localisation e6_origin; do \
			$(PY) experiments/$$e.py --data $(DATA)/$$c; \
		done; \
	done
