#!/usr/bin/env python3
"""
Búsqueda bibliográfica en OpenAlex para el estado del arte de la línea
"Diagnóstico de fallos en front-ends analógicos de ECG".

Qué hace:
  1. Lanza las búsquedas B1–B6 con el filtro title_and_abstract.search
     (operadores booleanos solo sobre título y resumen).
  2. Hace seguimiento de citas de los artículos clave:
       - hacia delante: trabajos que los citan
       - hacia atrás: trabajos que ellos citan
  3. Reconstruye el resumen a partir del índice invertido de OpenAlex.
  4. Guarda un JSON y un CSV por búsqueda, y un CSV maestro deduplicado
     que indica en qué búsquedas aparece cada trabajo.

Uso:
  pip install requests
  python openalex_sota.py

Configura MAILTO (recomendado por OpenAlex) y, si dispones de ella, API_KEY.
"""

import csv
import json
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

MAILTO = "tu_correo@ejemplo.com"   # cámbialo: da acceso al "polite pool"
API_KEY = ""                       # opcional; déjalo vacío si no tienes
OUT_DIR = Path("openalex_results")
MAX_RESULTS_PER_QUERY = 1000       # tope de seguridad por búsqueda
PER_PAGE = 200                     # máximo permitido por OpenAlex
SLEEP = 0.2                        # pausa entre peticiones (s)

BASE = "https://api.openalex.org"

SELECT_FIELDS = ",".join([
    "id", "doi", "title", "publication_year", "type",
    "primary_location", "cited_by_count", "abstract_inverted_index",
    "referenced_works",
])

# Búsquedas. Importante: las cadenas no pueden contener comas,
# porque OpenAlex usa la coma para separar filtros.
QUERIES = {
    "B1_componente_frontend_biomedico": {
        "q": '("fault diagnosis" OR "fault detection" OR "fault isolation" OR '
             '"self-test" OR "built-in test") AND ("ECG" OR "electrocardiogram" OR '
             '"biopotential" OR "EMG" OR "EEG") AND ("analog front-end" OR '
             '"front end" OR "amplifier" OR "acquisition circuit" OR '
             '"instrumentation amplifier")',
        "years": None,
    },
    "B2_nivel_senal_sensores_ecg": {
        "q": '("sensor fault" OR "sensor failure" OR "faulty sensor" OR '
             '"lead-off" OR "electrode fault" OR "fault diagnosis") AND '
             '("ECG" OR "electrocardiogram") AND ("machine learning" OR '
             '"deep learning" OR "classification")',
        "years": None,
    },
    "B3_autodiagnostico_equipos_medicos": {
        "q": '("self-test" OR "self-diagnosis" OR "built-in self-test" OR '
             '"predictive maintenance" OR "fault diagnosis") AND '
             '("medical device" OR "medical equipment" OR "patient monitor" OR '
             '"electrocardiograph")',
        "years": None,
    },
    "B4_gravedad_por_especificaciones": {
        "q": '("specification-based test" OR "specification-oriented test" OR '
             '"alternate test" OR "performance prediction" OR "parametric yield" OR '
             '"specification-based fault") AND ("analog circuit" OR "analog test" OR '
             '"mixed-signal")',
        "years": (2000, 2026),
    },
    "B5_electrodo_vs_circuito": {
        "q": '("electrode-skin impedance" OR "electrode impedance" OR '
             '"lead-off detection" OR "contact impedance") AND ("ECG" OR '
             '"biopotential") AND ("detection" OR "monitoring" OR "diagnosis")',
        "years": None,
    },
    "B6_circuitos_referencia_transferencia": {
        "q": '"analog circuit" AND "fault diagnosis" AND ("transfer learning" OR '
             '"domain adaptation" OR "benchmark circuit" OR "cross-circuit" OR '
             '"simulation-to-real" OR "sim-to-real")',
        "years": (2018, 2026),
    },
}

# Artículos clave para el seguimiento de citas (por DOI).
# Añade aquí el DOI del artículo de Dieste-Velasco en Integration (2025)
# cuando lo tengas.
SEED_DOIS = {
    "FauDigPro_2022": "10.1109/icmiam56779.2022.10146898",
    "Chen_2025_TIM": "10.1109/tim.2025.3586376",
    "DiesteVelasco_2021_Mathematics": "10.3390/math9243247",
    "DiesteVelasco_2024_AEJ": "10.1016/j.aej.2024.01.054",
}

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

session = requests.Session()


def _params(extra: dict) -> dict:
    p = dict(extra)
    if MAILTO:
        p["mailto"] = MAILTO
    if API_KEY:
        p["api_key"] = API_KEY
    return p


def get_json(url: str, params: dict, retries: int = 5) -> dict:
    """GET con reintentos y espera exponencial ante 429/5xx."""
    for attempt in range(retries):
        r = session.get(url, params=_params(params), timeout=60)
        if r.status_code == 200:
            time.sleep(SLEEP)
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            wait = 2 ** attempt
            print(f"  HTTP {r.status_code}; reintento en {wait}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    raise RuntimeError(f"Fallo persistente en {url}")


def reconstruct_abstract(inv: dict | None) -> str:
    """Reconstruye el resumen desde abstract_inverted_index."""
    if not inv:
        return ""
    positions = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def simplify(work: dict) -> dict:
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return {
        "openalex_id": work.get("id", "").replace("https://openalex.org/", ""),
        "doi": (work.get("doi") or "").replace("https://doi.org/", ""),
        "title": work.get("title") or "",
        "year": work.get("publication_year"),
        "type": work.get("type"),
        "venue": src.get("display_name") or "",
        "cited_by": work.get("cited_by_count", 0),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "n_references": len(work.get("referenced_works") or []),
    }


def paged_works(filter_str: str, limit: int) -> list[dict]:
    """Recorre resultados con paginación por cursor."""
    works, cursor = [], "*"
    while cursor and len(works) < limit:
        data = get_json(f"{BASE}/works", {
            "filter": filter_str,
            "per-page": PER_PAGE,
            "cursor": cursor,
            "select": SELECT_FIELDS,
        })
        if cursor == "*":
            print(f"  total en OpenAlex: {data['meta'].get('count')}")
        works.extend(data.get("results", []))
        cursor = data["meta"].get("next_cursor")
        if not data.get("results"):
            break
    return works[:limit]


def save(name: str, rows: list[dict]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with open(OUT_DIR / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"  guardados {len(rows)} registros -> {name}.json / .csv")


# ---------------------------------------------------------------------------
# 1. Búsquedas B1–B6
# ---------------------------------------------------------------------------

def run_queries() -> dict[str, list[dict]]:
    results = {}
    for name, spec in QUERIES.items():
        print(f"\n[{name}]")
        f = f"title_and_abstract.search:{spec['q']}"
        if spec["years"]:
            y0, y1 = spec["years"]
            f += f",publication_year:{y0}-{y1}"
        rows = [simplify(w) for w in paged_works(f, MAX_RESULTS_PER_QUERY)]
        rows.sort(key=lambda r: -(r["cited_by"] or 0))
        save(name, rows)
        results[name] = rows
    return results


# ---------------------------------------------------------------------------
# 2. Seguimiento de citas
# ---------------------------------------------------------------------------

def resolve_doi(doi: str) -> dict | None:
    try:
        return get_json(f"{BASE}/works/https://doi.org/{doi}",
                        {"select": SELECT_FIELDS})
    except RuntimeError as e:
        print(f"  no se pudo resolver {doi}: {e}")
        return None


def fetch_by_ids(ids: list[str]) -> list[dict]:
    """Recupera trabajos por lotes de 50 identificadores."""
    out = []
    for i in range(0, len(ids), 50):
        chunk = [x.replace("https://openalex.org/", "") for x in ids[i:i + 50]]
        data = get_json(f"{BASE}/works", {
            "filter": "openalex_id:" + "|".join(chunk),
            "per-page": 50,
            "select": SELECT_FIELDS,
        })
        out.extend(data.get("results", []))
    return out


def run_citation_chasing() -> dict[str, list[dict]]:
    results = {}
    for label, doi in SEED_DOIS.items():
        print(f"\n[citas: {label}]")
        seed = resolve_doi(doi)
        if not seed:
            continue
        wid = seed["id"].replace("https://openalex.org/", "")
        print(f"  {wid}: {seed.get('title')}")

        forward = [simplify(w) for w in paged_works(f"cites:{wid}", MAX_RESULTS_PER_QUERY)]
        save(f"CITAN_A_{label}", forward)
        results[f"CITAN_A_{label}"] = forward

        refs = seed.get("referenced_works") or []
        backward = [simplify(w) for w in fetch_by_ids(refs)] if refs else []
        save(f"CITADOS_POR_{label}", backward)
        results[f"CITADOS_POR_{label}"] = backward
    return results


# ---------------------------------------------------------------------------
# 3. Fichero maestro deduplicado
# ---------------------------------------------------------------------------

def build_master(all_results: dict[str, list[dict]]) -> None:
    master: dict[str, dict] = {}
    for source, rows in all_results.items():
        for r in rows:
            key = r["openalex_id"]
            if key not in master:
                master[key] = dict(r, sources=source)
            else:
                master[key]["sources"] += f";{source}"
    rows = sorted(master.values(),
                  key=lambda r: (-r["sources"].count(";"), -(r["cited_by"] or 0)))
    save("MAESTRO_deduplicado", rows)
    n_abs = sum(1 for r in rows if r["abstract"])
    print(f"\nMaestro: {len(rows)} trabajos únicos, {n_abs} con resumen")


if __name__ == "__main__":
    all_results = {}
    all_results.update(run_queries())
    all_results.update(run_citation_chasing())
    build_master(all_results)
    print("\nListo. Resultados en", OUT_DIR.resolve())
