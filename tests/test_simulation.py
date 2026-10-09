"""End-to-end tests against ngspice: nominal behaviour, specifications, faults, dataset."""

import numpy as np
import pytest
from spicefault import Dataset

from ecgfd.dataset import experiment, generate, load_cases, relabel
from ecgfd.features import feature_sets
from ecgfd.measurement import apply_measurement_model
from ecgfd.selftest import pulse_waveform
from ecgfd.specs import specifications

pytestmark = pytest.mark.ngspice

NOMINAL_GAIN = {"integrated": 278, "reference": 414}
# resistor from the gain-stage feedback node to the reference
GAIN_RESISTOR = {"integrated": "R11", "reference": "R17"}


@pytest.fixture(scope="module")
def study(cfg):
    return experiment(cfg)


@pytest.fixture(scope="module")
def nominal(study):
    return study.nominal()


def compliant(observed: dict, cfg: dict) -> bool:
    specs = observed["bench"].measurements
    return all(spec.met(specs[spec.name]) for spec in specifications(cfg))


def test_nominal_circuit_meets_every_specification(nominal, cfg):
    specs = nominal["bench"].measurements
    assert specs["spec_gain_error"] == 0.0  # the nominal gain is that of this circuit
    assert compliant(nominal, cfg)
    # 3 mV x 100 ms through a 3.9 s high-pass: 3 mV * (1 - exp(-0.1 / 3.9)) = 76 uV
    assert specs["spec_impulse_offset_uv"] == pytest.approx(76, abs=3)
    # 185 Hz Butterworth low-pass seen at 150 Hz
    assert specs["spec_resp_min_hf"] == pytest.approx(0.83, abs=0.02)
    # 620 kOhm against the two 10 MOhm bias resistors
    assert specs["spec_zin_drop"] == pytest.approx(0.03, abs=0.005)
    assert specs["spec_input_range_mv"] > 5.0


def test_nominal_self_test_measurements(nominal, cfg):
    adc = cfg["measurement"]["adc"]
    mid = 0.5 * (adc["vmin"] + adc["vmax"])
    gain = NOMINAL_GAIN[cfg["circuit"]]
    features = nominal["service"].measurements
    pulse = pulse_waveform(cfg)(nominal["service"].result)
    assert features["dc_out"] == pytest.approx(mid, abs=1e-3)
    assert len(pulse) == 1000
    assert np.abs(pulse[:95] - mid).max() < 1e-3  # quiet before the pulse
    assert pulse.max() - mid == pytest.approx(1e-3 * gain, rel=0.05)
    # the lead-off current sees both electrodes and protection resistors, about 124 kOhm
    assert features["zlo_mag_10"] / gain == pytest.approx(124e3, rel=0.1)


def test_circuit_fault_breaks_specs_but_electrode_fault_does_not(study, nominal, cfg):
    # shorting the gain-stage resistor to the reference saturates the output
    assert not compliant(study.nominal(f"{GAIN_RESISTOR[cfg['circuit']]}:short"), cfg)

    # a detached electrode is seen by the self-test, yet the circuit remains compliant
    detached = next(
        fault
        for fault in study.faults
        if fault.fault_type == "electrode_off" and fault.tags["component"] == "la"
    )
    lead_off, reference = study.nominal(detached), nominal["service"].measurements
    assert lead_off["service"].measurements["acd_mag_10"] < 0.1 * reference["acd_mag_10"]
    assert lead_off["service"].measurements["zlo_mag_10"] > 10 * reference["zlo_mag_10"]
    assert compliant(lead_off, cfg)


def test_small_deviation_can_remain_compliant(study, cfg):
    """H2: a component 5 % out of nominal is a "fault" by percentage, not by function."""
    assert compliant(study.nominal("R1:parametric:+0.05"), cfg)


def test_generate_and_load_dataset(cfg, tmp_path):
    small = {**cfg, "dataset": {"n_healthy": 3, "n_per_fault": 0}}
    generate(small, tmp_path / "a", jobs=2, progress=False)
    cases, waveforms, stored = load_cases(tmp_path / "a")
    assert len(cases) == 3 and waveforms.shape == (3, 1000)
    assert cases["sim_ok"].all() and (cases["fault_id"] == "healthy").all()
    assert cases["compliant"].all() and (cases["origin"] == "").all()
    assert stored["circuit"] == cfg["circuit"]
    assert stored["specs"]["nominal_gain"] == pytest.approx(NOMINAL_GAIN[cfg["circuit"]], rel=0.01)
    data = Dataset(tmp_path / "a")
    assert data.verify() == []
    # the pulse response is kept for the rows in service only
    on_bench = (data.samples["condition"] == "bench").to_numpy()
    assert np.isnan(data.waveforms[on_bench]).all() and len(data.netlist(1)) > 0

    measured, wav = apply_measurement_model(cases, waveforms, stored, np.random.default_rng(0))
    for features in feature_sets(stored).values():
        assert measured[features].notna().all().all()
    assert wav.shape == waveforms.shape

    # same seed, same data, whatever the number of jobs and the chunking
    generate(small, tmp_path / "b", jobs=1, progress=False, chunk=2)
    again, waveforms2, _ = load_cases(tmp_path / "b")
    np.testing.assert_array_equal(waveforms, waveforms2)
    assert cases["acd_mag_10"].equals(again["acd_mag_10"])
    assert cases["spec_cmrr_db"].equals(again["spec_cmrr_db"])


def test_relabel_applies_new_limits_without_simulating(cfg, tmp_path):
    small = {**cfg, "dataset": {"n_healthy": 3, "n_per_fault": 0}}
    generate(small, tmp_path, progress=False)
    cases = relabel(tmp_path, {**cfg["specs"], "noise_max_uvpp": 0.1})
    assert not cases["compliant"].any()
    assert (cases["violated"] == "spec_noise_uvpp").all()
    _, _, stored = load_cases(tmp_path)
    assert stored["specs"]["noise_max_uvpp"] == 0.1 and "nominal_gain" in stored["specs"]
    # the files still match their fingerprints, and the change is on record
    data = Dataset(tmp_path)
    assert data.verify() == []
    assert len(data.manifest.summary["history"]) == 2
    assert {s.name: s.maximum for s in data.specifications}["spec_noise_uvpp"] == 0.1

    with pytest.raises(ValueError):  # a different test set-up needs new simulations
        relabel(tmp_path, {**cfg["specs"], "electrode_offset": 0.1})
