import os

from models import FeatureSet
from seeders.utils import print_status


def seed_feature_sets(db):
    feature_sets_data = [
        {
            "feature_type": "tfidf",
            "data_type": "text",
            "hf_repo_id": os.getenv("HF_TFIDF_REPO"),
            "current_version": "v1",
        },
        {
            "feature_type": "vit",
            "data_type": "image",
            "hf_repo_id": os.getenv("HF_VIT_REPO"),
            "current_version": "v1",
        },
    ]

    counter = 0
    for fs in feature_sets_data:
        existing = (
            db.query(FeatureSet)
            .filter(FeatureSet.feature_type == fs["feature_type"])
            .one_or_none()
        )

        if existing:
            # Optional: keep DB in sync if repo/version changed
            existing.hf_repo_id = fs["hf_repo_id"]
            existing.current_version = fs["current_version"]
            existing.data_type = fs["data_type"]
        else:
            db.add(FeatureSet(**fs))
            counter += 1

    print_status("feature_sets", counter)
