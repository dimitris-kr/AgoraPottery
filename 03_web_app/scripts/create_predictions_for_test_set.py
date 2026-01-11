from database import SessionLocal
from models import PotteryItemInTrainingRun, PotteryItem, TrainingRun, ChronologyPrediction, ModelVersion, Model
from services import get_feature_types, select_model, download_image_tmp, extract_features, load_model, \
    load_target_decoder, predict_single, create_prediction_record


def get_current_training_run(db):
    return db.query(TrainingRun).filter_by(is_current=True).one_or_none()


def get_test_pottery_items(db, training_run_id):
    return (
        db.query(PotteryItem)
        .join(PotteryItemInTrainingRun)
        .filter(
            PotteryItemInTrainingRun.training_run_id == training_run_id,
            PotteryItemInTrainingRun.split == "test",
        )
        .all()
    )


def create_prediction_for_item(db, item, task):
    text = item.description
    image_repo_path = item.image_path

    # Determine Feature Types and Model
    feature_types = get_feature_types(text, image_repo_path)
    db_model, db_model_version = select_model(db, task, feature_types)

    try:
        image_tmp_path = download_image_tmp(image_repo_path) if image_repo_path else None
    except Exception:
        image_tmp_path = None

    # Extract Features
    features = extract_features(db, feature_types, text, image_tmp_path)
    feature_list = [feature_tensor for feature_tensor in features.values()]

    # Load Model
    model = load_model(db_model.hf_repo_id, db_model_version.version)

    decoder = load_target_decoder(db_model.hf_repo_id, db_model_version.version, task)

    # Predict
    prediction, breakdown = predict_single(task, model, feature_list, decoder)

    # Save Prediction
    prediction_record = create_prediction_record(db, task, text, image_repo_path, prediction, breakdown, db_model_version, status="validated")

    prediction_record.pottery_item_id = item.id

    db.add(prediction_record)
    db.flush()

    if image_tmp_path:
        image_tmp_path.unlink()

def run_batch_predictions():
    db = SessionLocal()

    try:
        training_run = get_current_training_run(db)

        test_items = get_test_pottery_items(db, training_run.id)

        for item in test_items:
            for task in ["classification", "regression"]:
                exists = (
                    db.query(ChronologyPrediction)
                    .filter_by(
                        pottery_item_id=item.id,
                    )
                    .join(ModelVersion)
                    .join(Model)
                    .filter(Model.task.has(name=task))
                    .first()
                )

                if exists:
                    continue

                create_prediction_for_item(db, item, task)

        db.commit()
        print("✅ Test set predictions created")

    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_batch_predictions()