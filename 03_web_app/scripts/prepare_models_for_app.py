import json
import shutil
from pathlib import Path

import joblib
import torch

# SOURCE (notebooks output)
SRC_ROOT = Path("../../02_chronology_prediction/NN")

# DESTINATION (web app)
DST_ROOT = Path("../uploaders/ml_models")

ENCODER_PATH = Path("../../data/targets/classification/y_encoder.pkl")
SCALER_PATH = Path("../../data/targets/regression/y_scaler.pkl")

MODEL_MAP = {
    "classification": [
        ("pottery_chronology_classifier_tfidf.pt", "agora_pottery_chronology_classifier_tfidf", "tfidf"),
        ("pottery_chronology_classifier_vit.pt", "agora_pottery_chronology_classifier_vit", "vit"),
        ("pottery_chronology_classifier_tfidf_vit.pt", "agora_pottery_chronology_classifier_tfidf_vit", "tfidf + vit"),
    ],
    "regression": [
        ("pottery_chronology_regressor_tfidf.pt", "agora_pottery_chronology_regressor_tfidf", "tfidf"),
        ("pottery_chronology_regressor_vit.pt", "agora_pottery_chronology_regressor_vit", "vit"),
        ("pottery_chronology_regressor_tfidf_vit.pt", "agora_pottery_chronology_regressor_tfidf_vit", "tfidf + vit"),
    ],
}

VERSION = "v1"

def get_input_size(key):
    sizes = []
    for k in key.split(" + "):
        X = torch.load(f"../../data/features2/tensors/X_train_{k}.pt")
        sizes.append(X.shape[1])
    return sizes

def prepare():
    DST_ROOT.mkdir(exist_ok=True)

    y_encoder = joblib.load(ENCODER_PATH)
    y_scaler = joblib.load(SCALER_PATH)

    for task, models in MODEL_MAP.items():
        src_models_dir = SRC_ROOT / task / "models"
        best_params_path = src_models_dir / "best_params.json"

        with open(best_params_path) as f:
            best_params = json.load(f)

        for model_file, model_name, key in models:
            model_src = src_models_dir / model_file
            model_dst_dir = DST_ROOT / model_name / VERSION
            model_dst_dir.mkdir(parents=True, exist_ok=True)

            # Copy model
            shutil.copy(model_src, model_dst_dir / "model.pt")

            entry = best_params[key]

            input_sizes = get_input_size(key)

            output_size = (
                len(y_encoder.classes_) if task == "classification"
                else y_scaler.scale_.shape[0]
            )

            # Save encoder / scaler
            if task == "classification":
                shutil.copy(ENCODER_PATH, model_dst_dir / "y_encoder.pkl")
            else:
                shutil.copy(SCALER_PATH, model_dst_dir / "y_scaler.pkl")

            # config.json
            config = {
                **entry["params"],
                "input_sizes": input_sizes,
                "output_size": output_size,
                "chronology_target": "periods" if task == "classification" else "years",
            }

            with open(model_dst_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

            # metadata.json
            metadata = {
                "val_loss": entry["val_loss"],
                "train_loss": entry.get("train_loss"),
                "scores": entry.get("scores"),
                "time": entry.get("time"),
            }

            with open(model_dst_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"✔ Prepared {model_name}/{VERSION}")

if __name__ == "__main__":
    prepare()
