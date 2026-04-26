import os
import sys
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_FILE = PROJECT_ROOT / "db" / "database.db"
SUBMISSION_DATABASE_FILE = PROJECT_ROOT / "db" / "submission_database.db"
MODEL_FILE = PROJECT_ROOT / "models" / "property_ml_models.pkl"
FRONTEND_FILE = PROJECT_ROOT / "frontend" / "index.html"
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file() -> None:
    # Load a lightweight local .env without adding another dependency so the
    # same entrypoint works in a simple marker setup.
    if not ENV_FILE.exists():
        return

    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def print_startup_report() -> None:
    host = get_host()
    port = get_port()
    database_file = get_database_file_for_report()
    print("LonCastAI startup checks")
    print("-----------------------")
    print(f"Project folder: {PROJECT_ROOT}")
    print(f"Database: {'OK' if database_file.exists() else 'MISSING'} ({database_file})")
    print(f"ML model: {'OK' if MODEL_FILE.exists() else 'MISSING'} ({MODEL_FILE})")
    print(f"Frontend: {'OK' if FRONTEND_FILE.exists() else 'MISSING'} ({FRONTEND_FILE})")
    print(f"OS Maps API key: {'OK' if os.getenv('OS_MAPS_API_KEY') else 'not set - fallback map will be used'}")
    print(f"Server bind: {host}:{port}")

    if not database_file.exists():
        print("\nWarning: no project database file was found. The app can start, but searches will not work until the database is provided or rebuilt.")

    if not MODEL_FILE.exists():
        print("\nWarning: models/property_ml_models.pkl is missing. The app can start, but ML predictions will be unavailable until the model is trained.")

    if host in {"127.0.0.1", "localhost"}:
        print("\nOpen the app at: http://127.0.0.1:8000")
    else:
        print("\nThe app is ready for cloud or network access through the configured host/port.")
    print("Press Ctrl+C to stop.\n")


def get_port() -> int:
    try:
        return int(os.getenv("PORT", "8000"))
    except ValueError:
        return 8000


def get_host() -> str:
    return os.getenv("HOST", "0.0.0.0" if os.getenv("PORT") else "127.0.0.1")


def get_database_file_for_report() -> Path:
    # The startup report should reflect whichever bundled database the app will
    # actually use on the current machine.
    if DATABASE_FILE.exists():
        return DATABASE_FILE
    if SUBMISSION_DATABASE_FILE.exists():
        return SUBMISSION_DATABASE_FILE
    return DATABASE_FILE


def main() -> None:
    os.chdir(PROJECT_ROOT)
    load_env_file()
    os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    print_startup_report()
    uvicorn.run("backend.main:app", host=get_host(), port=get_port(), reload=False)


if __name__ == "__main__":
    main()
