"""Download the vendor SPICE models used for validation into models/ (not tracked by git).

The TI model is copyrighted and is not redistributed with this repository; it is
only needed by `scripts/validate_ina_model.py`.

    python scripts/fetch_vendor_models.py
"""

from __future__ import annotations

import io
import urllib.request
import zipfile

from ecgfd.config import REPO_ROOT

MODELS_DIR = REPO_ROOT / "models"
# INA333 PSpice model, TI literature number SBOM382
INA333_URL = "https://www.ti.com/lit/zip/sbom382"


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    request = urllib.request.Request(INA333_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    target = MODELS_DIR / "INA333.LIB"
    target.write_bytes(archive.read("INA333.LIB"))
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
