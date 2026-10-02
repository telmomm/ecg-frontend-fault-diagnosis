"""Measurement realism: ADC noise, quantisation and clipping.

The dataset stores noise-free simulation results. This model is applied when
building the feature matrices, so the noise level and the ADC resolution are
study parameters (experiment E4) and never require re-simulating.

Simplifications, to be revisited when the acquisition chain is specified:
- a DC reading is the average of `dc_averages` samples;
- a tone is estimated from `n_samples` samples, so the error of its amplitude has
  standard deviation noise_rms * sqrt(2 / n_samples), and an output tone larger
  than the ADC range is clipped to the range;
- quantisation is applied to waveforms and DC readings, not to tone estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import add_derived, pulse_features


def quantise(v: np.ndarray, adc: dict) -> np.ndarray:
    """Clip to the ADC range and round to the nearest code, returning volts."""
    vmin, vmax = float(adc["vmin"]), float(adc["vmax"])
    lsb = (vmax - vmin) / (2 ** int(adc["bits"]) - 1)
    return vmin + np.round((np.clip(v, vmin, vmax) - vmin) / lsb) * lsb


def apply_measurement_model(
    df: pd.DataFrame,
    waveforms: np.ndarray,
    cfg: dict,
    rng: np.random.Generator,
    adc: dict | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (features as measured, waveforms as measured).

    The returned frame keeps every label/metadata column of `df`, replaces the raw
    features by their measured values and adds the derived ones (CMRR, pulse features).
    `adc` overrides `measurement.adc`, e.g. to sweep noise or resolution.
    """
    mcfg = cfg["measurement"]
    adc = {**mcfg["adc"], **(adc or {})}
    sigma = float(adc["noise_rms"])
    full_scale = 0.5 * (float(adc["vmax"]) - float(adc["vmin"]))  # largest tone amplitude
    out = df.copy()
    n = len(df)

    wav = quantise(waveforms + rng.normal(0.0, sigma, waveforms.shape), adc)

    sigma_dc = sigma / np.sqrt(float(adc["dc_averages"]))
    for node, spec in mcfg["dc_nodes"].items():
        scale = float(spec["scale"])
        v = df[f"dc_{node}"].to_numpy() * scale + rng.normal(0.0, sigma_dc, n)
        out[f"dc_{node}"] = quantise(v, adc) / scale

    # tones: (feature prefix, stimulus amplitude, frequencies, has a phase feature)
    ac, lead_off = mcfg["ac"], mcfg["lead_off"]
    sigma_tone = sigma * np.sqrt(2.0 / float(ac["n_samples"]))
    tones = (
        ("acd", float(ac["diff_amplitude"]), ac["freqs"], True),
        ("acc", float(ac["cm_amplitude"]), ac["freqs"], False),
        ("zlo", float(lead_off["current"]), lead_off["freqs"], False),
    )
    for prefix, amplitude, freqs, has_phase in tones:
        for f in freqs:
            col = f"{prefix}_mag_{f:g}"
            tone = np.minimum(df[col].to_numpy() * amplitude, full_scale)
            out[col] = np.abs(tone + rng.normal(0.0, sigma_tone, n)) / amplitude
            if has_phase:
                # phase error of a tone estimate is about sigma / amplitude radians
                err = np.degrees(sigma_tone / np.maximum(tone, sigma_tone))
                phase = df[f"{prefix}_ph_{f:g}"].to_numpy()
                out[f"{prefix}_ph_{f:g}"] = phase + rng.normal(0.0, 1.0, n) * err

    out = add_derived(out, cfg)
    pf = pulse_features(wav, cfg)
    pf.index = out.index
    return pd.concat([out, pf], axis=1), wav.astype(np.float32)
