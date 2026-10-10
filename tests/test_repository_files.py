import importlib.util
from pathlib import Path


def test_main_notebook_exists() -> None:
    assert Path("notebooks/FIV1_monolith.ipynb").is_file()


def test_src_package_is_discoverable() -> None:
    assert importlib.util.find_spec("src") is not None
