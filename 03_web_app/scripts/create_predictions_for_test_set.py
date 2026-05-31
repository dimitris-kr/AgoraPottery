from database import SessionLocal
from models import PotteryItemInTrainingRun, PotteryItem, TrainingRun, ChronologyPrediction, ModelVersion, Model
from seeders.config import DATE_CHRONOLOGY_PREDICTIONS, WINDOW_CHRONOLOGY_PREDICTIONS
from seeders.utils import get_spread_timestamp
from services import get_feature_types, select_model, download_image_tmp, extract_features, load_model, \
    load_target_decoder, predict_single, create_prediction_record

TASKS = ["classification", "regression"]
prediction_count = {task:{} for task in TASKS}

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

def get_input_type(text, image):
    input_types = []
    if text:
        input_types.append("text")
    if image:
        input_types.append("image")
    return "+".join(input_types)

def create_prediction_for_item(db, item, task, timestamp):
    text = item.description
    image_repo_path = item.image_path

    try:
        image_tmp_path = download_image_tmp(image_repo_path) if image_repo_path else None
    except Exception:
        image_tmp_path = None

    # Determine Feature Types and Model
    feature_types = get_feature_types(text, image_tmp_path)
    db_model, db_model_version = select_model(db, task, feature_types)

    # Extract Features
    features = extract_features(db, feature_types, text, image_tmp_path)
    feature_list = [feature_tensor for feature_tensor in features.values()]

    # Load Model
    model = load_model(db_model.hf_repo_id, db_model_version.version)

    decoder = load_target_decoder(db_model.hf_repo_id, db_model_version.version, task)

    # Predict
    prediction, breakdown = predict_single(task, model, feature_list, decoder)

    # Save Prediction
    prediction_record = create_prediction_record(db, task, text, image_repo_path, prediction, breakdown,
                                                 db_model_version, status="validated")

    prediction_record.pottery_item_id = item.id

    # Demo predictions get fixed historical timestamp (real dev timeline)
    # Live predictions get the default func.now() .
    # The caller passes an incrementing timestamp so the list view sorts naturally.
    prediction_record.created_at = timestamp
    prediction_record.updated_at = timestamp

    db.add(prediction_record)
    db.flush()

    if image_tmp_path:
        image_tmp_path.unlink()

    # Count
    count_key = (get_input_type(text, image_repo_path), db_model.name, db_model_version.version)
    if count_key in prediction_count[task]:
        prediction_count[task][count_key] += 1
    else:
        prediction_count[task][count_key] = 1


def run_batch_predictions():
    db = SessionLocal()

    try:
        print("✨ Starting creating test set predictions...")
        training_run = get_current_training_run(db)

        test_items = get_test_pottery_items(db, training_run.id)

        items_to_predict = []
        for item in test_items:
            for task in TASKS:
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

                items_to_predict.append((item, task))

        num_of_predictions = len(items_to_predict)
        for prediction_idx, (item, task) in enumerate(items_to_predict):
            timestamp = get_spread_timestamp(DATE_CHRONOLOGY_PREDICTIONS, WINDOW_CHRONOLOGY_PREDICTIONS, prediction_idx,
                                             num_of_predictions)
            create_prediction_for_item(db, item, task, timestamp)

        db.commit()

        print("✅ Test set predictions created:")
        print(f"    {num_of_predictions} total predictions:")
        for task in TASKS:
            print(f"        {sum(prediction_count[task].values())} {task} predictions:")
            for key, count in prediction_count[task].items():
                print(f"            {count} predictions with input: {key[0]} and model: {key[1]} {key[2]}")


    except Exception as error:
        db.rollback()
        print("⛔ Error during prediction creation: ", error)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_batch_predictions()
