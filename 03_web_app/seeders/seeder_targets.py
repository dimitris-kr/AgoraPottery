from models import Target
from seeders.utils import print_status


def seed_targets(db):
    targets = [
        "Start Year",
        "Year Range",
        "Historical Period",
    ]

    counter = 0
    for name in targets:
        exists = db.query(Target).filter_by(name=name).first()
        if not exists:
            db.add(Target(name=name))
            counter += 1

    print_status("targets", counter)