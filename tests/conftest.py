import pytest
from spicefault.simulation.ngspice import ngspice_path

from ecgfd.circuit import CIRCUITS
from ecgfd.config import DEFAULT_CONFIG, load_config


@pytest.fixture(scope="session", params=CIRCUITS)
def cfg(request) -> dict:
    """The default configuration, once per circuit."""
    return load_config(DEFAULT_CONFIG, request.param)


def pytest_collection_modifyitems(config, items):
    if ngspice_path() is not None:
        return
    skip = pytest.mark.skip(reason="ngspice not installed")
    for item in items:
        if "ngspice" in item.keywords:
            item.add_marker(skip)
