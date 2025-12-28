from models import DataSource
from seeders.utils import print_status


def seed_data_sources(db):
    sources = [
        "Original Dataset",
        "Data Upload",
        "Prediction Upload",
    ]

    counter = 0
    for description in sources:
        exists = db.query(DataSource).filter_by(description=description).first()
        if not exists:
            db.add(DataSource(description=description))
            counter += 1

    print_status("data_sources", counter)