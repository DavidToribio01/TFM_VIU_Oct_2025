"""Pruebas del empaquetado de los notebooks y de las rutas compartidas."""
from pathlib import Path
import json

from bias_detector.config import PROJECT_ROOT, DATA_DIR, MODEL_DATA_DIR


def test_default_paths_relative_to_repo():
    assert (PROJECT_ROOT / "src" / "bias_detector").is_dir()
    assert DATA_DIR.is_absolute()
    assert MODEL_DATA_DIR.is_absolute()


def test_adapted_colab_notebooks_do_not_mount_drive():
    for nb in (PROJECT_ROOT / "notebooks" / "colab").glob("*.ipynb"):
        document = json.loads(nb.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell["source"])
            for cell in document["cells"]
            if cell["cell_type"] == "code"
        )
        assert "drive.mount(" not in source, nb.name
        assert "/content/drive/" not in source, nb.name


def test_adapted_vs_notebooks_have_no_private_absolute_paths():
    for nb in (PROJECT_ROOT / "notebooks" / "visual_studio").glob("*.ipynb"):
        source = nb.read_text(encoding="utf-8")
        assert "C:\\\\Users" not in source, nb.name
