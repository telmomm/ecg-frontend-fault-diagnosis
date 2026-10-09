"""Check the behavioural INA model against the TI INA333 macromodel, at block level.

Both models are placed in the same bench (3.3 V supply, mid-supply reference, the
gain resistors of the `integrated` circuit) and compared on gain, bandwidth,
common-mode rejection, offset and output swing. The table goes to
results/ina_validation.csv.

The vendor model is used only here. Inside the complete front-end it needs about
40 s per case in ngspice and converges to a wrong operating point, so the dataset
is generated with the behavioural model.

    python scripts/fetch_vendor_models.py
    python scripts/validate_ina_model.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from spicefault import SimulationConfig, Simulator
from spicefault.measurements import interp_response

from ecgfd.circuit import load_circuit
from ecgfd.config import REPO_ROOT, load_config

VENDOR_LIB = REPO_ROOT / "models" / "INA333.LIB"
OUT = ("v(out)",)
ANALYSES = (
    ("op", OUT),
    "alter @vd[acmag]=1",
    ("ac dec 50 1 1e6", OUT),
    "alter @vd[acmag]=0",
    "alter @vcm[acmag]=1",
    ("ac lin 1 50 50", OUT),
    "alter vd dc=1",  # overdrive: output against the upper rail, then the lower one
    ("op", OUT),
    "alter vd dc=-1",
    ("op", OUT),
)


def bench(model_lines: list[str], instance: str, rg: float, vcc: float) -> str:
    return "\n".join(
        [
            "* INA bench",
            *model_lines,
            f"Vcc vcc 0 dc {vcc}",
            "Vee vee 0 dc 0",
            f"Vref ref 0 dc {vcc / 2}",
            f"Vcm cm 0 dc {vcc / 2} ac 0",
            "Vd inp cm dc 0 ac 0",
            "Vn inn cm dc 0",
            f"R5 rgp mid {rg}",
            f"R6 mid rgn {rg}",
            instance,
            "Rload out ref 100k",
            ".end",
            "",
        ]
    )


def characterise(netlist: str, vcc: float, spiceinit: str | None) -> dict[str, float]:
    result = Simulator().run(netlist, SimulationConfig(analyses=ANALYSES, spiceinit=spiceinit))
    if not result.ok:
        raise RuntimeError(f"{result.status.value}: {result.message}")
    op, ac, cm, high, low = result.plots
    freq, h = ac["frequency"].real, ac["v(out)"]
    gain = abs(interp_response(freq, h, 10.0))
    return {
        "gain_10hz": gain,
        "bandwidth_khz": float(freq[np.abs(h) >= gain / np.sqrt(2)][-1]) / 1e3,
        "cmrr_50hz_db": float(20 * np.log10(gain / abs(cm["v(out)"][0]))),
        "offset_rti_uv": 1e6 * (float(op["v(out)"][0].real) - vcc / 2) / gain,
        "swing_high_mv_from_rail": 1e3 * (vcc - float(high["v(out)"][0].real)),
        "swing_low_mv_from_rail": 1e3 * float(low["v(out)"][0].real),
    }


def main() -> None:
    circuit = load_circuit(load_config(circuit="integrated"))
    text = circuit.to_netlist()
    models = text[text.index(".subckt") : text.rindex(".ends ina") + len(".ends ina")]
    ina = circuit.component("XU1").parameters
    parameters = " ".join(f"{name}={value}" for name, value in ina.items())
    vcc, rg = circuit.component("Vcc").parameters["dc"], circuit.component("R5").parameters["value"]
    behavioural = bench(
        [models], f"XU1 inp inn rgp rgn ref out vcc vee ina {parameters}", rg, vcc
    )
    rows = {"behavioural": characterise(behavioural, vcc, None)}
    if VENDOR_LIB.exists():
        vendor = bench(
            [f'.include "{VENDOR_LIB}"'], "XU1 inp inn vcc vee out ref rgp rgn INA333", rg, vcc
        )
        # the TI library is written in the PSpice dialect
        rows["ti_macromodel"] = characterise(vendor, vcc, "set ngbehavior=psa")
    else:
        print(f"{VENDOR_LIB} not found: run scripts/fetch_vendor_models.py to compare")
    table = pd.DataFrame(rows)
    out = REPO_ROOT / "results" / "ina_validation.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out)
    print(table.to_string(float_format=lambda v: f"{v:.4g}"))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
