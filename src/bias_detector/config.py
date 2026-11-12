"""Rutas portables para notebooks y módulos (sin Google Drive)."""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _path(env_var: str, default: Path) -> Path:
    return Path(os.environ.get(env_var, str(default))).expanduser().resolve()

DATA_DIR = _path("TFM_DATA_DIR", PROJECT_ROOT / "data")
MODEL_DATA_DIR = _path("TFM_MODEL_DATA_DIR", DATA_DIR / "model_data")
MODEL_DIR = _path("TFM_MODEL_DIR", PROJECT_ROOT / "artifacts" / "model_v3")
PROMPTS_DIR = _path("TFM_PROMPTS_DIR", PROJECT_ROOT / "prompts")
RESULTS_DIR = _path("TFM_RESULTS_DIR", PROJECT_ROOT / "results")
ANNOTATION_DIR = _path("TFM_ANNOTATION_DIR", DATA_DIR / "annotation")

def ensure_output_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def require_file(path: Path, purpose: str = "archivo") -> Path:
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Falta {purpose}: {path}. Coloca los datos en data/ o configura la ruta TFM_* correspondiente."
        )
    return Path(path)

# Rutas reales del proyecto de Visual Studio, utilizadas también desde Colab.
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELING_DIR = DATA_DIR / "modeling"
PILOT_DIR = DATA_DIR / "pilot"
REDACTION_ANALYSIS_DIR = DATA_DIR / "redaction_analysis"
ANNOTATION_RESULTS_DIR = DATA_DIR / "results"
MASS_ANNOTATION_DIR = ANNOTATION_RESULTS_DIR / "mass_annotation"
FINAL_ANNOTATION_DIR = MASS_ANNOTATION_DIR / "final"
LUNA_RESULTS_DIR = MASS_ANNOTATION_DIR / "LUNA"
TERRA_RESULTS_DIR = MASS_ANNOTATION_DIR / "terra"
PILOT_RESULTS_DIR = ANNOTATION_RESULTS_DIR / "pilot"
MICRO_PILOT_RESULTS_DIR = ANNOTATION_RESULTS_DIR / "micro_pilot"

# Fuentes canónicas comunes a los notebooks de Visual Studio y Colab.
CORPUS_DATA = MODELING_DIR / "modeling_dataset_v0_2.pkl"
FINAL_ANNOTATIONS_DATA = FINAL_ANNOTATION_DIR / "final_bias_annotations_v0_2.pkl"
