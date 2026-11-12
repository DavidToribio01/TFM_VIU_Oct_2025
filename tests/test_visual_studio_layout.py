"""Comprueba la correspondencia de rutas sin requerir descargar los datos."""
from bias_detector import config as cfg

def test_canonical_visual_studio_sources():
    assert cfg.CORPUS_DATA == cfg.DATA_DIR / "modeling" / "modeling_dataset_v0_2.pkl"
    assert cfg.FINAL_ANNOTATIONS_DATA == cfg.DATA_DIR / "results" / "mass_annotation" / "final" / "final_bias_annotations_v0_2.pkl"
    assert cfg.PILOT_DIR == cfg.DATA_DIR / "pilot"
    assert cfg.LUNA_RESULTS_DIR == cfg.DATA_DIR / "results" / "mass_annotation" / "LUNA"
    assert cfg.TERRA_RESULTS_DIR == cfg.DATA_DIR / "results" / "mass_annotation" / "terra"

def test_colab_experiment_sources_share_canonical_locations():
    import json
    for name in ("training_experiments.ipynb", "explainability.ipynb"):
        notebook = json.loads((cfg.PROJECT_ROOT / "notebooks" / "colab" / name).read_text())
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        assert "CANONICAL_CORPUS_DATA" in source
        assert "FINAL_ANNOTATIONS_DATA" in source
        assert "DATA_DIR / 'training'" not in source
