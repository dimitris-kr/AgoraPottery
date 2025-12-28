from database import SessionLocal, engine, Base
from seeder_users import seed_users
from seeder_data_sources import seed_data_sources
from seeder_historical_periods import seed_historical_periods
from seeders.seeder_targets import seed_targets
from seeders.seeder_tasks import seed_tasks

if __name__ == "__main__":
    print("🌱 Running all seeders...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        seed_users(db)

        seed_data_sources(db)
        seed_historical_periods(db)

        seed_tasks(db)
        seed_targets(db)

        db.commit()
        print("✅ Seeding completed successfully!")
    except Exception as error:
        db.rollback()
        print("⛔ Error during seeding: ", error)
    finally:
        db.close()
