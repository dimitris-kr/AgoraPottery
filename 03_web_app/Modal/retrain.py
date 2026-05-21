# ──────────────────────────────────────────────
# GET DEVICE
# ──────────────────────────────────────────────

def _get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# HF DOWNLOAD HELPERS
# ──────────────────────────────────────────────

def _download_image(hf_path: str, hf_images_repo: str, hf_token: str):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=hf_images_repo,
        filename=hf_path,
        repo_type="dataset",
        token=hf_token,
    )


def download_config(repo_id: str, version: str, hf_token: str) -> dict:
    """Download config.json (hyperparameters) for a model version."""
    from huggingface_hub import hf_hub_download
    import json
    path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{version}/config.json",
        repo_type="model",
        token=hf_token,
    )
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────

# TF-IDF — fresh vocabulary per version (responsive to new tokens in new data)

def refit_tfidf_vectorizer(data_train: list[dict], max_features: int = 300):
    """Re-fit a TF-IDF vectorizer on the new training set."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Initialize
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
    )

    # Fit
    descriptions = [item.get("description") or "" for item in data_train]
    vectorizer.fit(descriptions)

    return vectorizer


def extract_tfidf_features(data: list[dict], vectorizer):
    """Extract TF-IDF tensors from item descriptions using a fitted vectorizer."""
    import torch
    descriptions = [item.get("description") or "" for item in data]
    X = vectorizer.transform(descriptions)
    return torch.tensor(X.toarray(), dtype=torch.float32)


# VIT - module-level cache so the model loads once per process/container.

_VIT: dict = {"extractor": None, "model": None, "device": None}


def _get_vit():
    if _VIT["model"] is None:
        from transformers import ViTImageProcessor, ViTModel
        device = _get_device()

        _VIT["extractor"] = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        _VIT["model"] = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(device).eval()
        _VIT["device"] = device

    return _VIT["extractor"], _VIT["model"], _VIT["device"]


def extract_vit_features(items: list[dict], hf_images_repo: str, hf_token: str):
    """
        Extract ViT embeddings for all item images.
        Items without an image_path use a zero-vector placeholder.
    """
    import torch
    from PIL import Image

    vit_extractor, vit_model, device = _get_vit()

    features = []
    for item in items:
        image_path = item.get("image_path")

        if not isinstance(image_path, str) or len(image_path) == 0:
            features.append(torch.zeros(768).to(device))
            continue

        local_path = _download_image(image_path, hf_images_repo, hf_token)

        image = Image.open(local_path).convert("RGB")
        inputs = vit_extractor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = vit_model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        features.append(embedding)

    return torch.stack(features)


# ──────────────────────────────────────────────
# TARGET BUILDING
# ──────────────────────────────────────────────

def get_regression_targets(items: list[dict]):
    return [[it["start_year"], it["year_range"]] for it in items]


def refit_y_scaler(y_train: list[list[float]]):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(y_train)
    return scaler


def scale_y(y: list[list[float]], scaler):
    import torch
    y = scaler.transform(y)
    return torch.tensor(y, dtype=torch.float32)


def get_classification_targets(items: list[dict]):
    return [it["historical_period"] for it in items]


def refit_y_encoder(y_train: list[str]):
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(y_train)
    return encoder


def encode_y(y: list[str], encoder):
    import torch
    y = encoder.transform(y)
    return torch.tensor(y, dtype=torch.long)


# ──────────────────────────────────────────────
# PotteryChronologyPredictor MODEL CLASS IMPORT SHIM
# ──────────────────────────────────────────────

def _import_predictor():
    """
    Import PotteryChronologyPredictor — tries local path first, then the
    Modal-container path where the file is mounted via add_local_file.
    """
    try:
        from ML.PotteryChronologyPredictor import (
            PotteryChronologyPredictor,
            PotteryDataset,
            activation_funcs,
            train,
        )
    except ImportError:
        import sys
        sys.path.insert(0, "/app")
        from PotteryChronologyPredictor import (
            PotteryChronologyPredictor,
            PotteryDataset,
            activation_funcs,
            train,
        )
    return PotteryChronologyPredictor, PotteryDataset, activation_funcs, train


# ──────────────────────────────────────────────
# SINGLE MODEL TRAINING
# ──────────────────────────────────────────────

# Get Data Dimensions
def _get_dimensions(X, y, feature_types, task, y_encoder=None):
    # input sizes follow the canonical feature order
    X_dim = [X["train"][ft].shape[1] for ft in feature_types]

    if task == "classification":
        # CLASSIFICATION → dims = number of classes
        if y_encoder is None:
            raise ValueError("y_encoder is required for classification tasks")
        y_dim = len(y_encoder.classes_)
    else:
        # REGRESSION -> dims = number of continuous variables
        y_dim = y["train"].shape[1] if y["train"].dim() > 1 else 1

    return X_dim, y_dim


def train_single_model(
        # *,
        X,
        y,
        task: str,
        config: dict,
        y_scaler=None,
        y_encoder=None,
) -> tuple:
    """
        Train one model.

        Inputs:
            X = {
                'train': {feature_type_1: torch.Tensor[N, dim1], ...},
                'val': {feature_type_1: torch.Tensor, ...},
            },
            y = {
                'train': torch.Tensor(classification -> [N, 1], regression -> [N, 2]),
                'val': torch.Tensor,
            },
            task = 'classification' | 'regression',

        Returns:
            model           — trained PotteryChronologyPredictor
            metadata        — train()'s return dict (train_loss, val_loss, scores, time)
            updated_config  — v1 config dict with input_sizes + output_size overridden
                            to reflect v2's actual architecture. Upload THIS dict as
                            the new config.json — NOT the original `config` argument.

        chronology_target is derived from `task` (DB source of truth), not config.

        Architecture is re-derived from the current data each call:
            - input_sizes : from X_train shapes (TF-IDF vocab may differ from previous version)
            - output_size :
                -- classification: from y_encoder.classes_ (robust pipeline to new historical periods
                                    in the training data)
                -- regression: from y_train cols
    """
    from torch.utils.data import DataLoader

    PotteryChronologyPredictor, PotteryDataset, activation_funcs, train = _import_predictor()

    device = _get_device()

    # Canonical feature order — MUST match predict-time ordering.
    # Sorted alphabetically (tfidf, vit) so encoder[i] is always bound to the
    # same feature, regardless of how the caller built the X dict.
    feature_types = sorted(X["train"].keys())

    # Dimensions  (re-derived from current data, not inherited from previous version config)
    X_dim, y_dim = _get_dimensions(X, y, feature_types, task, y_encoder)

    # Torch Datasets and Dataloaders
    datasets = {
        subset: PotteryDataset([X[subset][ft] for ft in feature_types], y[subset])
        for subset in X.keys()
    }
    loaders = {
        subset: DataLoader(dataset, batch_size=64, shuffle=True if subset == "train" else False)
        for subset, dataset in datasets.items()
    }

    # Build Model
    model = PotteryChronologyPredictor(
        X_dim,
        y_dim,
        config["hidden_size"],
        device=device,
        activation=activation_funcs[config["activation"]],
        dropout=config["dropout"],
        blocks=config["blocks"],
        hidden_size_pattern=config["hidden_size_pattern"],
        chronology_target="periods" if task == "classification" else "years",
    )

    # Train Model
    model, metadata = train(
        model,
        loaders["train"],
        loaders["val"],
        y_scaler=y_scaler,
        lr=config["lr"],
    )

    # Updated config — input_sizes + output_size reflect new version's actual architecture.
    # feature_types records the canonical order so predict-time can reproduce it.
    updated_config = {
        **config,
        "feature_types": feature_types,
        "input_sizes": X_dim,
        "output_size": y_dim,
    }

    return model, metadata, updated_config


# ──────────────────────────────────────────────
# HF UPLOAD HELPERS
# ──────────────────────────────────────────────

# HF artifact filenames — MUST match the download side (services/hf_service.py)
TFIDF_FILENAME     = "tfidf_vectorizer.joblib"
Y_ENCODER_FILENAME = "y_encoder.pkl"
Y_SCALER_FILENAME  = "y_scaler.pkl"
MODEL_FILENAME     = "model.pt"
CONFIG_FILENAME    = "config.json"
METADATA_FILENAME  = "metadata.json"


def upload_processor(
        repo_id: str,
        version: str,
        repo_type: str,
        processor,
        processor_filename: str,
        hf_token: str,
):
    """Upload a joblib-serializable object (tfidf vectorizer / y_encoder / y_scaler)."""
    import joblib
    from huggingface_hub import upload_file
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / processor_filename
        joblib.dump(processor, path)
        upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"{version}/{processor_filename}",
            repo_id=repo_id,
            repo_type=repo_type,
            token=hf_token,
        )


def upload_json(
        repo_id: str,
        version: str,
        repo_type: str,
        data: dict,
        data_filename: str,
        hf_token: str,
):
    """Upload a dict as a JSON file (config.json / metadata.json)."""
    from huggingface_hub import upload_file
    import tempfile
    from pathlib import Path
    import json

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / data_filename
        path.write_text(json.dumps(data, indent=2))
        upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"{version}/{data_filename}",
            repo_id=repo_id,
            repo_type=repo_type,
            token=hf_token,
        )


def upload_model(
        repo_id: str,
        version: str,
        model_state_dict,
        model_filename: str,
        hf_token: str,
        repo_type: str = "model",
):
    """Upload a torch state dict (model.pt)."""
    import torch
    from huggingface_hub import upload_file
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / model_filename
        torch.save(model_state_dict, path)
        upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"{version}/{model_filename}",
            repo_id=repo_id,
            repo_type=repo_type,
            token=hf_token,
        )


def upload_model_files(
        repo_id: str,
        version: str,
        model_state_dict,
        config: dict,
        metadata: dict,
        hf_token: str,
):
    """Uploads model.pt, config.json (the UPDATED one), and metadata.json."""

    upload_model(
        repo_id=repo_id,
        version=version,
        model_state_dict=model_state_dict,
        model_filename=MODEL_FILENAME,
        hf_token=hf_token,
    )

    upload_json(
        repo_id=repo_id,
        version=version,
        repo_type="model",
        data=config,
        data_filename=CONFIG_FILENAME,
        hf_token=hf_token,
    )

    upload_json(
        repo_id=repo_id,
        version=version,
        repo_type="model",
        data=metadata,
        data_filename=METADATA_FILENAME,
        hf_token=hf_token,
    )
