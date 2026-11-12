"""Ejemplo de análisis retrospectivo; requiere Longformer descargado."""
import argparse
from pathlib import Path
from bias_detector.config import MODEL_DIR, RESULTS_DIR
from bias_detector.pipeline import BiasDetectionPipeline, save_pipeline_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("conversation_file", type=Path, help="Texto [TURN n | USER/LLM]")
    parser.add_argument("--conversation-id", default="CONVERSATION_EXAMPLE")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "analysis.json")
    args = parser.parse_args()
    pipeline = BiasDetectionPipeline(experiment_dir=args.model_dir)
    result = pipeline.run(conversation_id=args.conversation_id, conversation_text=args.conversation_file.read_text(encoding="utf-8"))
    save_pipeline_result(result, args.output)
    print(f"Resultado guardado en: {args.output}")

if __name__ == "__main__":
    main()
