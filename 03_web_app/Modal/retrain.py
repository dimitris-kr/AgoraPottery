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
