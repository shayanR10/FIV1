from pathlib import Path


def test_main_notebook_exists() -> None:
    assert Path("FIV1.ipynb").is_file()
