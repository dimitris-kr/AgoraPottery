from models import HistoricalPeriod
from seeders.print_status import print_status


def seed_historical_periods(db):
    periods = [
        ("Geometric", -900, -700),
        ("Orientalizing", -700, -600),
        ("Archaic", -600, -480),
        ("Classical", -480, -323),
        ("Hellenistic", -323, -31),
        ("Roman", -31, 330),
        ("Late Roman / Early Byzantine", 330, 700),
    ]

    counter = 0
    for name, limit_lower, limit_upper in periods:
        exists = db.query(HistoricalPeriod).filter_by(name=name).first()
        if not exists:
            db.add(HistoricalPeriod(
                name=name,
                limit_lower=limit_lower,
                limit_upper=limit_upper
            ))
            counter += 1

    print_status("historical_periods", counter)