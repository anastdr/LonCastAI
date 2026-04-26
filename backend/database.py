import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


def get_database_url() -> str:
    # Prefer an explicit deployment DATABASE_URL, but fall back to the local
    # project databases so the same code works for GitHub submission and local runs.
    configured_url = os.getenv("DATABASE_URL")
    if configured_url:
        return configured_url

    main_database = Path("db/database.db")
    submission_database = Path("db/submission_database.db")

    if main_database.exists():
        return f"sqlite:///{main_database}"
    if submission_database.exists():
        return f"sqlite:///{submission_database}"

    return f"sqlite:///{main_database}"


DATABASE_URL = get_database_url()

# SQLite needs a thread-safety override for FastAPI's request lifecycle, while
# other database engines should keep their default SQLAlchemy connection options.
engine_options = {}
if DATABASE_URL.startswith("sqlite"):
    engine_options["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_options)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
