import numpy as np
from sklearn.model_selection import train_test_split

from models import PotteryItem, HistoricalPeriod, ChronologyLabel, TrainingRun, PotteryItemInTrainingRun
from seeders.config import PATH_DATA, TRAIN_SPLIT, STRATIFY_COLUMN, RANDOM_STATE, TEST_SPLIT, VAL_SPLIT
from seeders.utils import load_data, print_status

def seed_training_runs(db):

    # TRAINING RUN
    training_run = (
        db.query(TrainingRun)
        .filter(TrainingRun.is_current == True)
        .one_or_none()
    )

    counter = 0
    if not training_run:
        db.query(TrainingRun).update({TrainingRun.is_current: False})

        training_run = TrainingRun(
            split_strategy=f"{TRAIN_SPLIT}-{VAL_SPLIT}-{TEST_SPLIT}",
            random_state=RANDOM_STATE,
            is_current=True
        )
        db.add(training_run)
        db.flush()  # get training_run.id
        counter += 1

    print_status('training_runs', counter)

    training_run_id = training_run.id


    # POTTERY ITEMS IN TRAINING RUN

    data_full = load_data(PATH_DATA)

    set_names = ["train", "val", "test"]

    indices_full = np.arange(data_full.shape[0])
    indices = {}

    indices["train"], indices_val_test = train_test_split(
        indices_full,
        test_size=(1 - TRAIN_SPLIT),
        stratify=data_full[STRATIFY_COLUMN],
        random_state=RANDOM_STATE,
    )

    indices["val"], indices["test"] = train_test_split(
        indices_val_test,
        test_size=TEST_SPLIT / (TEST_SPLIT + VAL_SPLIT),
        random_state=RANDOM_STATE,
    )

    data = {}
    for set_name in set_names:
        data[set_name] = data_full.loc[indices[set_name]]

    pottery_map = {
        p.object_id: p.id
        for p in db.query(PotteryItem).all()
    }

    existing_pairs = {
        (r.training_run_id, r.pottery_item_id)
        for r in db.query(PotteryItemInTrainingRun).filter(
            PotteryItemInTrainingRun.training_run_id == training_run_id
        )
    }

    associations = []
    for set_name in set_names:
        for _, row in data[set_name].iterrows():
            pottery_item_id = pottery_map[row["Id"]]
            if pottery_item_id is None:
                continue

            if (training_run_id, pottery_item_id) in existing_pairs:
                continue

            associations.append({
                "training_run_id": training_run_id,
                "pottery_item_id": pottery_item_id,
                "split": set_name,
            })

    db.bulk_insert_mappings(PotteryItemInTrainingRun, associations)

    print_status('pottery_items_in_training_runs', len(associations))