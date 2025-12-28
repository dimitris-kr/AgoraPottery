import pandas as pd
from sqlalchemy.orm import Session

from database import SessionLocal
from models import PotteryItem, DataSource
from seeders.config import PATH_DATA
from seeders.utils import load_data, print_status


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

    db.bulk_insert_mappings(PotteryItem, items)

    print_status('pottery_items', len(items))
