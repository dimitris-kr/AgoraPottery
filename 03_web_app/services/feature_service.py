import joblib
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

from models import FeatureSet
from services import download_tfidf_vectorizer



def get_tfidf_vectorizer(db):
    fs = (
        db.query(FeatureSet)
        .filter(FeatureSet.feature_type=='tfidf')
        .one()
    )

    vectorizer_file = download_tfidf_vectorizer(fs.hf_repo_id, fs.current_version)

    tfidf_vectorizer = joblib.load(vectorizer_file)

    return tfidf_vectorizer

def extract_tfidf_features(text: str, vectorizer):
    if not text or not text.strip():
        raise ValueError("Text input is empty")

    X = vectorizer.transform([text])  # shape (1, n_features)
    X = torch.from_numpy(X.toarray()).float()
    return X

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_VIT_EXTRACTOR = None
_VIT_MODEL = None

def get_vit_components():
    global _VIT_EXTRACTOR, _VIT_MODEL

    if _VIT_EXTRACTOR is None or _VIT_MODEL is None:
        _VIT_EXTRACTOR = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        _VIT_MODEL = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        ).to(_DEVICE).eval()

    return _VIT_EXTRACTOR, _VIT_MODEL

def extract_vit_features_from_upload(image_file):
    extractor, model = get_vit_components()

    image = Image.open(image_file.file).convert("RGB")

    inputs = extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # mean pooling
    X = outputs.last_hidden_state.mean(dim=1)

    return X  # shape (1, 768)

def extract_features(db, feature_types, text, image):
    features = {}

    if "tfidf" in feature_types:
        tfidf_vectorizer = get_tfidf_vectorizer(db)
        features["tfidf"] = extract_tfidf_features(text, tfidf_vectorizer)

    if "vit" in feature_types:
        features["vit"] = extract_vit_features_from_upload(image)

    return features