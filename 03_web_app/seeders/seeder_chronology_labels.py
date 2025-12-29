from models import PotteryItem, HistoricalPeriod, ChronologyLabel
from seeders.config import PATH_DATA
from seeders.utils import load_data, print_status


def seed_chronology_labels(db):
    df = load_data(PATH_DATA)

    pottery_map = {
        p.object_id: p.id
        for p in db.query(PotteryItem).all()
    }

    period_map = {
        p.name: p.id
        for p in db.query(HistoricalPeriod).all()
    }

    items = []
    for _, row in df.iterrows():
        pottery_item_id = pottery_map[row["Id"]]
        if not pottery_item_id:
            continue

        exists = db.query(ChronologyLabel).filter_by(
            pottery_item_id=pottery_item_id
        ).first()

        if exists:
            continue

        items.append({
            "pottery_item_id": pottery_item_id,
            "historical_period_id": period_map[row["HistoricalPeriod"]],

            "start_year": row["StartYear"],
            "end_year": row["EndYear"],
            "midpoint_year": row["MidpointYear"],
            "year_range": row["YearRange"],
        })

    db.bulk_insert_mappings(ChronologyLabel, items)

    print_status('chronology_labels', len(items))