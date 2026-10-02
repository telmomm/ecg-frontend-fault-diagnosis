"""End-to-end tests against ngspice: nominal behaviour, specifications, faults, dataset."""

import numpy as np
import pytest

from ecgfd.circuit import nominal_instance
from ecgfd.dataset import generate, load_dataset
from ecgfd.faults import Fault
from ecgfd.features import feature_sets
from ecgfd.measurement import apply_measurement_model
from ecgfd.simulate import measure, scalar_features
from ecgfd.specs import compliance, measure_specs, with_nominal_gain

pytestmark = pytest.mark.ngspice

NOMINAL_GAIN = {"integrated": 501, "reference": 992}
# resistor from the gain-stage feedback node to the reference
GAIN_RESISTOR = {"integrated": "R11", "reference": "R17"}


@pytest.fixture(scope="module")
def spec_cfg(cfg):
    return with_nominal_gain(cfg)


@pytest.fixture(scope="module")
def nominal(cfg):
    return measure(nominal_instance(cfg), cfg)


def test_nominal_circuit_meets_every_specification(spec_cfg):
    assert spec_cfg["specs"]["nominal_gain"] == pytest.approx(
        NOMINAL_GAIN[spec_cfg["circuit"]], rel=0.01
    )
    specs = measure_specs(nominal_instance(spec_cfg), spec_cfg)
    assert compliance(specs, spec_cfg)["violated"] == ""
    assert specs["f_low"] == pytest.approx(0.49, abs=0.02)
    assert specs["f_high"] == pytest.approx(178, abs=4)


def test_nominal_self_test_measurements(nominal, cfg):
    adc = cfg["measurement"]["adc"]
    mid = 0.5 * (adc["vmin"] + adc["vmax"])
    gain = NOMINAL_GAIN[cfg["circuit"]]
    assert nominal.dc["out"] == pytest.approx(mid, abs=1e-3)
    assert len(nominal.pulse) == 1000
    assert np.abs(nominal.pulse[:95] - mid).max() < 1e-3  # quiet before the pulse
    assert nominal.pulse.max() - mid == pytest.approx(1e-3 * gain, rel=0.05)
    # the lead-off current sees both electrodes and protection resistors, about 124 kOhm
    z = scalar_features(nominal, cfg)["zlo_mag_10"] / gain
    assert z == pytest.approx(124e3, rel=0.1)


def test_circuit_fault_breaks_specs_but_electrode_fault_does_not(nominal, spec_cfg):
    base = nominal_instance(spec_cfg)
    ref = scalar_features(nominal, spec_cfg)

    # shorting the gain-stage resistor to the reference saturates the output
    shorted = Fault("short", GAIN_RESISTOR[spec_cfg["circuit"]]).apply(base, spec_cfg)
    assert not compliance(measure_specs(shorted, spec_cfg), spec_cfg)["compliant"]

    # a detached electrode is seen by the self-test, yet the circuit remains compliant
    lead_off = Fault("electrode_off", "la").apply(base, spec_cfg)
    features = scalar_features(measure(lead_off, spec_cfg), spec_cfg)
    assert features["acd_mag_10"] < 0.1 * ref["acd_mag_10"]
    assert features["zlo_mag_10"] > 10 * ref["zlo_mag_10"]
    assert compliance(measure_specs(lead_off, spec_cfg), spec_cfg)["compliant"]


def test_small_deviation_can_remain_compliant(spec_cfg):
    """H2: a component 5 % out of nominal is a "fault" by percentage, not by function."""
    base = nominal_instance(spec_cfg)
    shifted = Fault("parametric", "R1", 0.05).apply(base, spec_cfg)
    assert compliance(measure_specs(shifted, spec_cfg), spec_cfg)["compliant"]


def test_generate_and_load_dataset(cfg, tmp_path):
    small = {**cfg, "dataset": {"n_healthy": 3, "n_per_fault": 0}}
    generate(small, tmp_path, jobs=2, progress=False)
    df, waveforms, stored = load_dataset(tmp_path)
    assert len(df) == 3 and waveforms.shape == (3, 1000)
    assert df["sim_ok"].all() and not df["is_faulty"].any()
    assert df["compliant"].all() and (df["origin"] == "none").all()
    assert stored["circuit"] == cfg["circuit"] and "nominal_gain" in stored["specs"]

    measured, wav = apply_measurement_model(df, waveforms, stored, np.random.default_rng(0))
    for features in feature_sets(stored).values():
        assert measured[features].notna().all().all()
    assert wav.shape == waveforms.shape

    # same seed, same data
    generate(small, tmp_path / "again", jobs=1, progress=False)
    df2, waveforms2, _ = load_dataset(tmp_path / "again")
    np.testing.assert_array_equal(waveforms, waveforms2)
    assert df["acd_mag_10"].equals(df2["acd_mag_10"])
