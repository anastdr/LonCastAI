from sqlalchemy import text
from backend.database import engine

TABLE_NAME = "property_features"

with engine.connect() as conn:
    conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_NAME}"))
    conn.commit()

print(f"{TABLE_NAME} dropped successfully.")