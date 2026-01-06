import json

import joblib
import torch
from torch import nn

from ML import PotteryChronologyPredictor
from services import download_model, download_model_config, download_y_encoder, download_y_scaler

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
