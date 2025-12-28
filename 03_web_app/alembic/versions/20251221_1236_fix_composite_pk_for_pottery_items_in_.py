"""fix composite pk for pottery_items_in_feature_sets

Revision ID: 20251221_1236
Revises: 20251130_1232
Create Date: 2025-12-21 12:36:32.576901

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20251221_1236'
down_revision: Union[str, Sequence[str], None] = '20251130_1232'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop old primary key
    op.drop_constraint(
        "pottery_items_in_feature_sets_pkey",
        "pottery_items_in_feature_sets",
        type_="primary"
    )

    # Drop surrogate id column
    op.drop_column("pottery_items_in_feature_sets", "id")

    # Create composite primary key
    op.create_primary_key(
        "pottery_items_in_feature_sets_pkey",
        "pottery_items_in_feature_sets",
        ["pottery_item_id", "feature_set_id"]
    )


def downgrade() -> None:
    op.drop_constraint(
        "pottery_items_in_feature_sets_pkey",
        "pottery_items_in_feature_sets",
        type_="primary"
    )

    op.add_column(
        "pottery_items_in_feature_sets",
        sa.Column("id", sa.Integer(), primary_key=True)
    )
