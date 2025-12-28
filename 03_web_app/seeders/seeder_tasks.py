from models import Task
from seeders.utils import print_status


def seed_tasks(db):
    tasks = [
        "Classification",
        "Regression",
    ]

    counter = 0
    for name in tasks:
        exists = db.query(Task).filter_by(name=name).first()
        if not exists:
            db.add(Task(name=name))
            counter += 1

    print_status("tasks", counter)