from database import SessionLocal, engine, Base
from seeder_users import seed_users

if __name__ == "__main__":
    print("🌱 Running all seeders...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    seed_users(db)

    db.close()

    print("✅ Seeding complete!")