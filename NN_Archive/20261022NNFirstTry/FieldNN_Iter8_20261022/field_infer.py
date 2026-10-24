"""Reuse the full FDR inference/report-axis pipeline with Iter8 features."""
import importlib.util
from pathlib import Path
import features
import models

HERE = Path(__file__).resolve().parent
ITER3 = HERE.parent / 'FieldNN_Iter3_20261022'
ITER5 = HERE.parent / "FieldNN_Iter5_20261022"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    pipeline = load(ITER3 / 'field_eval.py', 'iter8_fdr_pipeline')
    decoder = load(ITER5 / 'decode.py', 'iter8_frozen_decoder')
    pipeline.build_model = models.build_model
    pipeline.preprocess = features.preprocess
    pipeline.decode_predictions = decoder.decode_predictions
    return pipeline.main()


if __name__ == '__main__':
    raise SystemExit(main())
