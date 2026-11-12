"""Recupera datos experimentales del ZIP original de Colab a data/ (ignorada por Git).

Ejemplo:
    python scripts/import_colab_archive.py '/ruta/Repositorio Colab.zip'

No descomprime archivos arbitrarios ni sobrescribe existentes por defecto.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def destination_for(name: str) -> Path | None:
    base = "entrenamiento/model_data/"
    if name.startswith(base):
        rel = Path(name[len(base):])
        if not rel.parts or any(p in {"..", ""} for p in rel.parts):
            return None
        return ROOT / "data" / "model_data" / rel
    prefix = "entrenamiento/"
    if name.startswith(prefix) and name.endswith(("modeling_dataset_v0_2.pkl", "final_bias_annotations_v0_2.pkl")):
        return (ROOT / "data" / "modeling" / Path(name).name if name.endswith("modeling_dataset_v0_2.pkl") else ROOT / "data" / "results" / "mass_annotation" / "final" / Path(name).name)
    if name in ("qwen_micro_pilot/micro_pilot_dataset.pkl", "qwen_micro_pilot/pilot_dataset.pkl"):
        return ROOT / "data" / "qwen_micro_pilot" / Path(name).name
    return None


def import_archive(archive: Path, *, dry_run: bool = False, overwrite: bool = False) -> tuple[int, int]:
    copied = skipped = 0
    with ZipFile(archive) as zf:
        for item in zf.infolist():
            if item.is_dir():
                continue
            dst = destination_for(item.filename)
            if dst is None:
                continue
            dst = dst.resolve()
            if not dst.is_relative_to((ROOT / "data").resolve()):
                raise ValueError(f"Ruta de destino no permitida: {dst}")
            if dst.exists() and not overwrite:
                skipped += 1
                continue
            if dry_run:
                print(f"{item.filename} -> {dst.relative_to(ROOT)} ({item.file_size:,} bytes)")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(item) as source, dst.open("wb") as target:
                    shutil.copyfileobj(source, target)
            copied += 1
    return copied, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zipfile", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    n, s = import_archive(args.zipfile, dry_run=args.dry_run, overwrite=args.overwrite)
    print(f"Archivos {'previstos' if args.dry_run else 'importados'}: {n}; existentes omitidos: {s}")
