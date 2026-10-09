"""Tests of the study definition that do not need ngspice.

What `spicefault` provides (netlist editing, distributions, campaign, splits,
ambiguity and sensitivity analysis) is tested there.
"""

import numpy as np
import pytest
from spicefault.experiments import sample_stream

from ecgfd.config import REPO_ROOT, load_config
from ecgfd.evaluation import centroid_separability, escape_rate, false_reject_rate
from ecgfd.faults import fault_catalogue, fault_universe
from ecgfd.population import ELECTRODES, front_end, load_circuit, population
from ecgfd.specs import SPEC_NAMES, specifications


@pytest.fixture(scope="module")
def circuit(cfg):
    return load_circuit(cfg)


def test_config_selects_circuit_and_extends(cfg):
    assert "circuits" not in cfg
    smoke = load_config(REPO_ROOT / "configs" / "smoke.yaml", cfg["circuit"])
    assert smoke["dataset"]["n_per_fault"] < cfg["dataset"]["n_per_fault"]
    assert smoke["faults"] == cfg["faults"]
    with pytest.raises(ValueError):
        load_config(circuit="no-such-circuit")


def test_circuit_size_matches_study_scope(circuit):
    passives, opamps, inas = front_end(circuit)
    assert 15 <= len(passives) <= 30
    assert 5 <= len(opamps) <= 6
    assert bool(inas) == (circuit.name == "integrated")
    # nothing in the netlist escapes the library, apart from the inside of the models
    assert all("subcircuit element" in problem for problem in circuit.check())


def test_population_is_reproducible_and_within_tolerance(circuit, cfg):
    healthy = population(circuit, cfg)

    def draw(*key):
        return healthy.sample(sample_stream(cfg["seed"], *key), circuit.netlist())

    a, b, c = draw(3, 7), draw(3, 7), draw(3, 8)
    assert a == b
    assert a.values != c.values
    nominal = circuit.parameters()
    for name in front_end(circuit)[0]:
        tolerance = cfg["tolerances"]["resistor" if name.startswith("R") else "capacitor"]
        assert abs(a.values[name, "value"] / nominal[name, "value"] - 1.0) <= tolerance


def test_electrodes_are_drawn_around_their_type(circuit, cfg):
    families, spread = cfg["electrodes"]["families"], cfg["electrodes"]["spread"]
    healthy = population(circuit, cfg)
    seen = set()
    for seed in range(60):
        drawn = healthy.sample(np.random.default_rng(seed), circuit.netlist())
        seen.add(drawn.labels["electrode_kind"])
        medians = families[drawn.labels["electrode_type"]][drawn.labels["electrode_kind"]]
        for name in ELECTRODES:
            for key, median in medians.items():
                value = drawn.values[f"{key.capitalize()}_{name}", "value"]
                assert median / spread <= value <= median * spread
    assert seen == {kind for family in families.values() for kind in family}


def test_catalogue_covers_circuit_and_electrodes(circuit, cfg):
    faults = fault_catalogue(circuit, cfg)
    ids = [fault.fault_id for fault in faults]
    assert len(set(ids)) == len(ids) and "healthy" not in ids
    assert {fault.tags["origin"] for fault in faults} == {"circuit", "electrode"}
    passives, opamps, inas = front_end(circuit)
    located = {fault.tags["component"] for fault in faults if fault.tags["origin"] == "circuit"}
    assert located == {*passives, *(name[1:] for name in opamps + inas)}
    # every fault but the balanced electrode pair comes from a rule of the universe
    universe = fault_universe(circuit, cfg)
    assert universe.coverage() == 1.0
    assert len(faults) == len(universe) + len(cfg["faults"]["electrode"]["high_z_factor"])
    assert universe.coverage_matrix().loc["R1", ["open", "short", "parametric"]].tolist() == [
        1,
        1,
        len(cfg["faults"]["parametric"]),
    ]


def test_faults_are_injected_into_a_copy(circuit, cfg):
    faults = {fault.fault_id: fault for fault in fault_catalogue(circuit, cfg)}
    nominal = circuit.parameters()
    text = circuit.to_netlist()

    def injected(fault):
        netlist = circuit.netlist()
        fault.apply(netlist)
        return netlist

    opened = injected(faults["R7:open"])
    assert opened.value("Rser_R7") == cfg["faults"]["r_open"]
    assert opened.nodes("R7")[1] == "R7_x"
    shifted = injected(faults["C5:parametric:+0.2"])
    assert np.isclose(shifted.value("C5"), 1.2 * nominal["C5", "value"])
    degraded = injected(faults["C4:cap_degradation:+0.5"])
    assert np.isclose(degraded.value("C4"), 0.5 * nominal["C4", "value"])
    assert degraded.value("Rser_C4") == 100.0

    both = injected(
        next(f for f in faults.values() if f.tags["component"] == "la+ra" and f.magnitude == 5)
    )
    for name in ("la", "ra"):
        assert both.value(f"Rd_{name}") == 5.0 * nominal[f"Rd_{name}", "value"]
        assert both.value(f"Cd_{name}") == nominal[f"Cd_{name}", "value"] / 5.0
    assert both.value("Rd_rl") == nominal["Rd_rl", "value"]
    assert circuit.to_netlist() == text


def test_specifications_take_their_limits_from_the_configuration(cfg):
    specs = specifications(cfg)
    assert tuple(s.name for s in specs) == SPEC_NAMES
    limits = {s.name: s.maximum if s.minimum is None else s.minimum for s in specs}
    assert limits["spec_cmrr_db"] == cfg["specs"]["cmrr_min_db"]
    # a value exactly at its limit passes; beyond it, or not computed, it does not
    assert all(s.met(limits[s.name]) for s in specs)
    by_name = {s.name: s for s in specs}
    assert not by_name["spec_resp_min_hf"].met(limits["spec_resp_min_hf"] - 0.1)
    assert not by_name["spec_noise_uvpp"].met(limits["spec_noise_uvpp"] + 0.1)
    assert not by_name["spec_noise_uvpp"].met(np.nan)


def test_decision_rates():
    is_bad = np.array([0, 0, 0, 0, 1, 1])
    pred = np.array([1, 0, 0, 0, 0, 1])
    assert false_reject_rate(is_bad, pred) == 0.25
    assert escape_rate(is_bad, pred) == 0.5


def test_centroid_separability_grows_with_class_distance():
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 200)
    noise = rng.normal(size=(400, 2))
    near = noise + labels[:, None] * 0.5
    far = noise + labels[:, None] * 5.0
    assert centroid_separability(far, labels) > centroid_separability(near, labels)
