from database import SessionLocal, engine, Base
from models.User import User
from services import hash_password

def seed_users():
    """Creates an admin user if not already existing."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    fields = {
        "username": "admin",
        "password": "agor@p0ttery25!"
    }

    # Check if exists
    existing_user = db.query(User).filter(User.username == fields["username"]).first()
    if existing_user:
        print(f"User '{fields["username"]}' already exists.")
        db.close()
        return

    # Hash password
    fields["hashed_password"] = hash_password(fields["password"])
    fields.pop("password")

    # Create object
    user = User(**fields)

    # Insert to DB
    db.add(user)
    db.commit()
    db.close()

    print(f"✅ Superuser '{fields["username"]}' created successfully!")
