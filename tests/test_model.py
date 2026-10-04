"""Tests that do not need ngspice: config, sampling, faults, labels, measurement model, metrics."""

import numpy as np
import pandas as pd
import pytest

from ecgfd.ambiguity import (
    a_priori_groups,
    ambiguity_groups,
    collinear_groups,
    component_groups,
    confusable_components,
    envelope_detection,
    fault_dictionary,
    separability,
)
from ecgfd.ambiguity import (
    testability_rank as visible_rank,  # alias: pytest would collect test*
)
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
from ecgfd.features import feature_sets, measurement_groups
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


def test_electrodes_are_drawn_around_their_type(cfg):
    families, spread = cfg["electrodes"]["families"], cfg["electrodes"]["spread"]
    seen = set()
    for seed in range(60):
        inst = sample_instance(cfg, np.random.default_rng(seed))
        seen.add(inst.electrode_kind)
        spec = families[inst.electrode_type][inst.electrode_kind]
        centres = spec.get("subject_rd", [spec["rd"]])  # one measured subject, or the median
        for electrode in inst.electrodes.values():
            assert spec["rs"] / spread <= electrode["rs"] <= spec["rs"] * spread
            assert any(c / spread <= electrode["rd"] <= c * spread for c in centres)
            # the time constant of the type is kept, within the spread of both parameters
            tau = electrode["rd"] * electrode["cd"] / (spec["rd"] * spec["cd"])
            assert 1 / spread**2 <= tau <= spread**2
    assert seen == {kind for family in families.values() for kind in family}


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

    bad = dict(at_limit, resp_min_hf=limits["resp_min_hf"][1] - 0.1, noise_uvpp=float("nan"))
    labels = compliance(bad, cfg)
    assert not labels["compliant"]
    assert labels["violated"] == "resp_min_hf,noise_uvpp"
    assert labels["ok_gain_error"] and not labels["ok_resp_min_hf"]


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


def test_confusable_components_are_not_transitive():
    # A1 ~ B1 and B2 ~ C1, but A and C share nothing
    names = ["A1", "B1", "B2", "C1"]
    d = pd.DataFrame(9.0, index=names, columns=names)
    for a, b in (("A1", "B1"), ("B2", "C1")):
        d.loc[a, b] = d.loc[b, a] = 1.0
    component_of = {"A1": "A", "B1": "B", "B2": "B", "C1": "C"}
    assert confusable_components(d, component_of) == {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
    assert component_groups(d, component_of) == [["A", "B", "C"]]


def test_small_deviation_testability():
    # z per 1 %: A and B move the same feature in proportion, C another, D almost nothing
    z = pd.DataFrame(
        {"A": [1.0, 0.0], "B": [-2.0, 0.0], "C": [0.0, 0.5], "D": [0.01, 0.0]}, index=["f1", "f2"]
    )
    groups, insensitive = collinear_groups(z, deviation_pct=10.0, threshold=3.0)
    assert insensitive == ["D"]
    assert sorted(map(sorted, groups)) == [["A", "B"], ["C"]]
    # two independent directions, both visible for a 10 % deviation
    assert visible_rank(z, deviation_pct=10.0, threshold=3.0) == 2
    assert visible_rank(z, deviation_pct=1.0, threshold=3.0) == 0


def test_robust_dictionary_sees_a_saturating_fault():
    """A few huge values must not hide that most cases of a fault moved far away."""
    rng = np.random.default_rng(0)
    healthy = 275 + rng.normal(0, 4, 500)
    fault = np.r_[rng.normal(0.03, 0.02, 190), np.full(10, 1650.0)]  # mostly near zero
    df = pd.DataFrame(
        {
            "condition": ["healthy"] * 500 + ["F"] * 200,
            "kind": ["healthy"] * 500 + ["short"] * 200,
            "g": np.r_[healthy, fault],
        }
    )
    centre, spread = fault_dictionary(df, ["g"])
    assert separability(centre, spread).loc["healthy", "F"] > 30
    detected, false_alarm = envelope_detection(df, ["g"], coverage=0.99)
    assert detected["F"] == 1.0 and false_alarm <= 0.02


def test_measurement_groups_cover_the_features_once(cfg):
    groups = measurement_groups(cfg)
    columns = [c for cols, _ in groups.values() for c in cols]
    assert len(columns) == len(set(columns))
    assert all(time > 0 for _, time in groups.values())
    # every feature of the sets is obtained from some group (CMRR from both tones of a frequency)
    available = set(columns) | {c.replace("acc_mag", "cmrr_db") for c in columns}
    for features in feature_sets(cfg).values():
        assert set(features) <= available
    # the slowest action is the lowest tone: two periods of it
    slowest = max(groups, key=lambda name: groups[name][1])
    assert slowest.endswith("_0.05") and groups[slowest][1] == 40.0


def test_a_priori_groups_follow_parallel_sensitivities():
    # A and B act alike, C differently, D is negligible and must not be grouped by chance
    z = pd.DataFrame(
        {"R1": [1.0, 0.0], "R2": [-2.0, 0.0], "C1": [0.0, 0.5], "C2": [1e-6, 0.0]},
        index=["f1", "f2"],
    )
    assert a_priori_groups(z) == {"R1": "R1+R2", "R2": "R1+R2", "C1": "C1", "C2": "C2"}
