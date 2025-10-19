from models.User import User
from services import hash_password

def seed_users(db):
    """Creates an admin user if not already existing."""

    fields = {
        "username": "admin",
        "password": "agor@p0ttery25!"
    }

    # Check if exists
    existing_user = db.query(User).filter(User.username == fields["username"]).first()
    if existing_user:
        print(f"User '{fields["username"]}' already exists.")
        return

    # Hash password
    fields["hashed_password"] = hash_password(fields["password"])
    fields.pop("password")

    # Create object
    user = User(**fields)

    # Insert to DB
    db.add(user)
    db.commit()

    print(f"✅ Superuser '{fields["username"]}' created successfully!")
