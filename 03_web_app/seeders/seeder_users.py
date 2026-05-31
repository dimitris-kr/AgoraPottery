import os

from models.User import User
from seeders.config import DATE_USERS
from seeders.utils import print_status
from services import hash_password

def seed_users(db):
    """Creates an admin user if not already existing.

    Credentials come from .env file - no hardcoded password in this seeder
    """

    password = os.getenv("ADMIN_PASSWORD")
    if not password:
        raise RuntimeError("ADMIN_PASSWORD is not set in the environment (.env)")

    fields = {
        "username": os.getenv("ADMIN_USERNAME", "admin"),
        "password": password,

        "created_at": DATE_USERS,
        "updated_at": DATE_USERS,
    }

    # Check if exists
    existing_user = db.query(User).filter(User.username == fields["username"]).first()
    if existing_user:
        print(f"❎ users: no additions")
        return

    # Hash password
    fields["hashed_password"] = hash_password(fields["password"])
    fields.pop("password")

    # Create object
    user = User(**fields)

    # Insert to DB
    db.add(user)

    print_status('users', 1)
