from database import SessionLocal, engine, Base
from seeders.seeder_chronology_labels import seed_chronology_labels
from seeders.seeder_feature_sets import seed_feature_sets
from seeders.seeder_models import seed_models
from seeders.seeder_training_runs import seed_training_runs
from seeders.seeder_users import seed_users
from seeders.seeder_data_sources import seed_data_sources
from seeders.seeder_historical_periods import seed_historical_periods
from seeders.seeder_pottery_items import seed_pottery_items
from seeders.seeder_targets import seed_targets
from seeders.seeder_tasks import seed_tasks

if __name__ == "__main__":
    print("🌱 Running all seeders...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # Flush after each seeder to make newly added rows available to the next seeders,
        # while keeping single commit at the end

        seed_users(db)
        db.flush()

        seed_data_sources(db)
        seed_historical_periods(db)
        seed_tasks(db)
        seed_targets(db)
        db.flush()

        seed_pottery_items(db)
        db.flush()

        seed_chronology_labels(db)
        db.flush()

        seed_training_runs(db)
        db.flush()

        seed_feature_sets(db)
        db.flush()

        seed_models(db)

        db.commit()
        print("✅ Seeding completed successfully!")
    except Exception as error:
        db.rollback()
        print("⛔ Error during seeding: ", error)
    finally:
        db.close()
