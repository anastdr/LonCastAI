from backend.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS epc_properties"))
    conn.commit()

print("epc_properties table dropped.")