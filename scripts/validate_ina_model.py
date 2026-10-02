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

from ecgfd.circuit import INTEGRATED, SUBCIRCUITS
from ecgfd.config import REPO_ROOT, load_config
from ecgfd.simulate import interp_response
from ecgfd.spice import RAW_NAME, run_deck

VENDOR_LIB = REPO_ROOT / "models" / "INA333.LIB"


def bench(model_lines: list[str], instance: str, cfg: dict) -> str:
    rg = INTEGRATED.passive("R5").value
    vcc = cfg["supply"]["vcc"]
    write = f"write {RAW_NAME} v(out)"
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
            ".control",
            "set noaskquit",
            "set appendwrite",
            "op",
            write,
            "alter @vd[acmag]=1",
            "ac dec 50 1 1e6",
            write,
            "alter @vd[acmag]=0",
            "alter @vcm[acmag]=1",
            "ac lin 1 50 50",
            write,
            "alter vd dc=1",  # overdrive: output against the upper rail, then the lower one
            "op",
            write,
            "alter vd dc=-1",
            "op",
            write,
            ".endc",
            ".end",
            "",
        ]
    )


def characterise(deck: str, vcc: float, spiceinit: str | None) -> dict[str, float]:
    op, ac, cm, high, low = run_deck(deck, spiceinit=spiceinit)
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
    cfg = load_config(circuit="integrated")
    ina = cfg["ina"]
    vcc = cfg["supply"]["vcc"]
    behavioural = bench(
        [SUBCIRCUITS],
        f"XU1 inp inn rgp rgn ref out vcc vee ina rfb={ina['rfb']} rdiff={ina['rdiff']} "
        f"cmrr={10 ** (ina['cmrr_db'] / 20)} aol={ina['aol']} gbw={ina['gbw']} "
        f"rout={ina['rout']} en={ina['en']} hr={ina['headroom']}",
        cfg,
    )
    rows = {"behavioural": characterise(behavioural, vcc, None)}
    if VENDOR_LIB.exists():
        vendor = bench(
            [f'.include "{VENDOR_LIB}"'], "XU1 inp inn vcc vee out ref rgp rgn INA333", cfg
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
