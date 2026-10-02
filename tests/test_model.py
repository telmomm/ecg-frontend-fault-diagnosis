"""Tests that do not need ngspice: config, sampling, faults, labels, measurement model, metrics."""

import numpy as np
import pandas as pd
import pytest

from ecgfd.ambiguity import ambiguity_groups
from ecgfd.circuit import build_netlist, get_circuit, nominal_instance
from ecgfd.config import REPO_ROOT, load_config
from ecgfd.dataset import build_tasks, sample_rng
from ecgfd.evaluation import (
    centroid_separability,
    escape_rate,
    false_reject_rate,
    magnitude_split,
    replica_split,
)
from ecgfd.faults import HEALTHY, Fault, fault_catalogue
from ecgfd.measurement import quantise
from ecgfd.sampling import passive_tolerance, sample_instance
from ecgfd.specs import SPEC_NAMES, compliance, spec_limits


def test_config_selects_circuit_and_extends(cfg):
    assert "circuits" not in cfg
    smoke = load_config(REPO_ROOT / "configs" / "smoke.yaml", cfg["circuit"])
    assert smoke["dataset"]["n_per_fault"] < cfg["dataset"]["n_per_fault"]
    assert smoke["faults"] == cfg["faults"]
    with pytest.raises(ValueError):
        load_config(circuit="no-such-circuit")


def test_circuit_size_matches_study_scope(cfg):
    circuit = get_circuit(cfg)
    assert 15 <= len(circuit.passives) <= 30
    assert len({p.name for p in circuit.passives}) == len(circuit.passives)
    assert bool(circuit.inas) == (circuit.name == "integrated")


def test_sampling_is_reproducible_and_within_tolerance(cfg):
    a = sample_instance(cfg, sample_rng(cfg, 3, 7))
    b = sample_instance(cfg, sample_rng(cfg, 3, 7))
    c = sample_instance(cfg, sample_rng(cfg, 3, 8))
    assert a == b
    assert a != c
    for p in get_circuit(cfg).passives:
        assert abs(a.passives[p.name] / p.value - 1.0) <= passive_tolerance(p.name, cfg)


def test_truncnorm_respects_tolerance(cfg):
    cfg = {**cfg, "tolerances": {**cfg["tolerances"], "distribution": "truncnorm"}}
    inst = sample_instance(cfg, np.random.default_rng(0))
    for p in get_circuit(cfg).passives:
        assert abs(inst.passives[p.name] / p.value - 1.0) <= passive_tolerance(p.name, cfg)


def test_electrodes_are_drawn_within_their_family(cfg):
    families = cfg["electrodes"]["families"]
    seen = set()
    for seed in range(20):
        inst = sample_instance(cfg, np.random.default_rng(seed))
        seen.add(inst.electrode_type)
        for electrode in inst.electrodes.values():
            for key, (lo, hi) in families[inst.electrode_type].items():
                assert lo <= electrode[key] <= hi
    assert seen == set(families)


def test_catalogue_ids_are_unique(cfg):
    faults = fault_catalogue(cfg)
    assert len({f.id for f in faults}) == len(faults)
    assert HEALTHY.id not in {f.id for f in faults}
    assert {f.origin for f in faults} == {"circuit", "electrode"}
    assert HEALTHY.origin == "none"
    d = cfg["dataset"]
    assert len(build_tasks(cfg)) == d["n_healthy"] + len(faults) * d["n_per_fault"]


def test_fault_injection_leaves_original_untouched(cfg):
    base = nominal_instance(cfg)
    opened = Fault("open", "R7").apply(base, cfg)
    assert base.series_r == {}
    assert opened.series_r == {"R7": cfg["faults"]["r_open"]}
    assert "Rser_R7" in build_netlist(opened, cfg, ["op"])

    shifted = Fault("parametric", "C5", 0.2).apply(base, cfg)
    assert np.isclose(shifted.passives["C5"], 1.2 * base.passives["C5"])

    degraded = Fault("cap_degradation", "C4", 0.5).apply(base, cfg)
    assert np.isclose(degraded.passives["C4"], 0.5 * base.passives["C4"])
    assert degraded.series_r["C4"] == 100.0

    both = Fault("electrode_high_z", "la+ra", 5.0).apply(base, cfg)
    assert both.electrodes["la"]["rd"] == 5.0 * base.electrodes["la"]["rd"]
    assert both.electrodes["rl"] == base.electrodes["rl"]


def test_compliance_labels(cfg):
    limits = spec_limits(cfg)
    assert set(limits) == set(SPEC_NAMES)
    # a value exactly at its limit passes
    at_limit = {name: limit for name, (_, limit) in limits.items()}
    assert compliance(at_limit, cfg)["compliant"]

    bad = dict(at_limit, f_high=limits["f_high"][1] - 1.0, noise_uvpp=float("nan"))
    labels = compliance(bad, cfg)
    assert not labels["compliant"]
    assert labels["violated"] == "f_high,noise_uvpp"
    assert labels["ok_gain_error"] and not labels["ok_f_high"]


def test_quantise_clips_and_rounds():
    adc = {"bits": 2, "vmin": -1.5, "vmax": 1.5}  # codes at -1.5, -0.5, 0.5, 1.5
    v = quantise(np.array([-9.0, -0.4, 0.1, 9.0]), adc)
    np.testing.assert_allclose(v, [-1.5, -0.5, 0.5, 1.5])


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


def _toy_labels() -> pd.DataFrame:
    rows = [("healthy", "healthy", 0.0)] * 10
    for level in (-0.2, -0.1, 0.1, 0.2):
        rows += [(f"R1:parametric:{level:+g}", "parametric", level)] * 10
    rows += [("R1:open", "open", 0.0)] * 10
    return pd.DataFrame(rows, columns=["condition", "kind", "level"])


def test_replica_split_is_a_partition_covering_every_condition():
    df = _toy_labels()
    train, test = replica_split(df, 0.3, seed=1)
    assert sorted(np.concatenate([train, test])) == list(range(len(df)))
    assert set(df.loc[test, "condition"]) == set(df["condition"])


def test_magnitude_split_holds_out_whole_magnitudes():
    df = _toy_labels()
    train, test = magnitude_split(df, test_levels=[-0.2, 0.2], seed=1)
    parametric = df[df["kind"] == "parametric"]
    assert set(parametric.level[parametric.index.isin(train)]) == {-0.1, 0.1}
    assert set(parametric.level[parametric.index.isin(test)]) == {-0.2, 0.2}
    assert "R1:open" in set(df.loc[train, "condition"]) & set(df.loc[test, "condition"])


def test_ambiguity_groups_merge_close_conditions():
    names = ["healthy", "a", "b", "c"]
    d = pd.DataFrame(
        [[0, 1, 9, 9], [1, 0, 9, 9], [9, 9, 0, 9], [9, 9, 9, 0]], index=names, columns=names
    )
    assert ambiguity_groups(d, threshold=3.0) == [["healthy", "a"], ["b"], ["c"]]
