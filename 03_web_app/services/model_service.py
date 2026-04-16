import json

import joblib
import torch
from fastapi import HTTPException
from torch import nn

from ML import PotteryChronologyPredictor
from models import Model, ModelVersion
from services import download_model, download_model_config, download_y_encoder, download_y_scaler, \
    download_model_metadata

_MODELS = {}
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DECODERS = {}

activation_funcs = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}

def build_model(config):
    model = PotteryChronologyPredictor(
        config["input_sizes"],
        config["output_size"],
        config["hidden_size"],
        device=_DEVICE,
        activation=activation_funcs[config["activation"]],
        dropout=config["dropout"],
        blocks=config["blocks"],
        hidden_size_pattern=config["hidden_size_pattern"],
        chronology_target=config["chronology_target"],
    )
    return model

def load_model(repo_id: str, version: str):
    key = f"{repo_id}:{version}"
    if key in _MODELS:
        return _MODELS[key]

    model_weights = download_model(repo_id, version)
    model_config = download_model_config(repo_id, version)

    with open(model_config) as f:
        config = json.load(f)

    model = build_model(config)
    state_dict = torch.load(model_weights, map_location=_DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(_DEVICE)

    _MODELS[key] = model

    return model


def load_target_decoder(repo_id: str, version: str, task: str):
    key = f"{repo_id}:{version}:{task}"

    if key in _DECODERS:
        return _DECODERS[key]

    if task == "classification":
        path = download_y_encoder(repo_id, version)
    else:
        path = download_y_scaler(repo_id, version)

    decoder = joblib.load(path)
    _DECODERS[key] = decoder
    return decoder

def validate_model_exists(model: Model | None):
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

def validate_model_version_exists(model_version: ModelVersion | None):
    if model_version is None:
        raise HTTPException(status_code=404, detail="Model Version not found")

def load_model_scores(repo_id: str, version: str, task: str):
    model_metadata = download_model_metadata(repo_id, version)

    with open(model_metadata) as f:
        metadata = json.load(f)

    return normalize_scores(metadata, task)


def normalize_scores(metadata: dict, task: str) -> list[dict]:
    scores = metadata.get("scores", {})

    result = []

    if task.lower() == "classification":
        # Single target
        target_scores = []

        for metric, values in scores.items():
            target_scores.append({
                "metric": metric,
                "value": values[0]
            })

        result.append({
            "target": "historical_period",
            "scores": target_scores
        })

    elif task.lower() == "regression":
        # Regression → 2 targets
        start_year_scores = []
        year_range_scores = []

        for metric, values in scores.items():
            if len(values) >= 1:
                start_year_scores.append({
                    "metric": metric,
                    "value": values[0]
                })

            if len(values) >= 2:
                year_range_scores.append({
                    "metric": metric,
                    "value": values[1]
                })

        result.append({
            "target": "start_year",
            "scores": start_year_scores
        })

        result.append({
            "target": "year_range",
            "scores": year_range_scores
        })

    return result