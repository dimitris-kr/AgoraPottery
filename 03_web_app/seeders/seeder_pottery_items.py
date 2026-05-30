import pandas as pd

from models import PotteryItem, DataSource
from seeders.config import PATH_DATA, DATE_POTTERY_ITEMS, WINDOW_POTTERY_ITEMS
from seeders.utils import load_data, print_status, get_spread_timestamp


def seed_pottery_items(db):
    df = load_data(PATH_DATA)

    source = db.query(DataSource).filter_by(
        description="Original Dataset"
    ).one()

    existing = {
        p.object_id
        for p in db.query(PotteryItem.object_id).all()
    }

    items = []

    for _, row in df.iterrows():
        if row["Id"] in existing:
            continue

        items.append({
            "data_source_id": source.id,
            "object_id": row["Id"] if not pd.isna(row['Id']) else None,
            "description": row["FullText"] if not pd.isna(row['FullText']) else None,
            "image_path": row["ImageFilename"] if not pd.isna(row['ImageFilename']) else None
        })

    # Spread created_at evenly across a time window on the import day, so list
    # views sort naturally instead of tying on one identical timestamp.
    num_of_items = len(items)
    for item_idx, item in enumerate(items):
        timestamp = get_spread_timestamp(DATE_POTTERY_ITEMS, WINDOW_POTTERY_ITEMS, item_idx, num_of_items)
        item["created_at"] = timestamp
        item["updated_at"] = timestamp

    db.bulk_insert_mappings(PotteryItem, items)

    print_status('pottery_items', len(items))
