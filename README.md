# LonCastAI

## What the software does

LonCastAI is a web-based decision-support system for short-term London property price prediction. It helps users search for a property, review an explainable price estimate, compare baseline and machine-learning outputs, and understand the local factors influencing the prediction.

## Project aim and intended users

The main aim of the project is to support short-term property pricing decisions in the London market through transparent and explainable predictions. The system is intended primarily for buyers, sellers, investors, and estate agents who want a simple way to review likely price levels without carrying out the full market research process manually.

## Core features implemented

- Property search by postcode with live postcode suggestions
- Address suggestion list for matching properties within the selected postcode
- Baseline property valuation using processed local property and sold-price data
- Machine-learning prediction using trained KNN and Random Forest models
- Mixed prediction that combines the baseline and ML estimate
- Explanation panel with pricing drivers, local context, property size, and last sold price information where available
- Confidence-style prediction summary to support interpretation and trust
- Interactive London map with postcode focus and nearby amenity context
- Amenity-aware context using station, school, crime, and house price index features
- Contact form that can send messages to the project email through the backend
- Responsive frontend for desktop and laptop use

## Alignment with the project brief

This submission follows the brief by prioritising the core predictive function over secondary features. The main implemented focus is:

- short-term property price estimation
- explainable output rather than black-box prediction only
- minimal user input through postcode search and property selection
- user-facing clarity through readable text, map context, and prediction breakdowns

The original brief also discussed broader dashboard-style insight features, postcode comparison, cloud deployment, and a PostgreSQL-backed architecture. For the final submission, these were simplified where needed to keep the project feasible, reliable, and runnable on another machine for marking.

## Project structure

- `backend/` FastAPI backend, prediction logic, routes, and database access
- `frontend/` HTML, CSS, JavaScript, branding, map, and search interface
- `db/database.db` processed SQLite database used by the app
- `models/property_ml_models.pkl` trained ML model artifact used by the app
- `scripts/` optional data processing and ML training scripts
- `requirements.txt` Python dependencies
- `start_mac.command` helper launcher for macOS
- `start_windows.bat` helper launcher for Windows

## Dependencies and install steps

### Software requirements

- Python 3.11 or newer recommended
- `pip`
- Internet connection recommended for map tiles

### Python packages

Install all required packages with:

```bash
pip install -r requirements.txt
```

The project does not require Node.js.

## Setup and run instructions

These instructions are the recommended marker workflow. If the marker follows them and the submitted folder includes the processed database and trained model, the system should run without rebuilding raw data.

For GitHub submission, the recommended database file is:

- `db/submission_database.db`

This compact database is suitable for the selected project postcodes and is small enough to include in a normal repository. The application automatically uses `db/submission_database.db` if `db/database.db` is not present.

### Before starting

Make sure the submitted project folder includes these files:

- `db/submission_database.db` or `db/database.db`
- `models/property_ml_models.pkl`
- `frontend/`
- `backend/`
- `requirements.txt`
- `run_app.py`

If both database files are missing, or if `models/property_ml_models.pkl` is missing, the application can still open, but property search or ML predictions will not work correctly.

### macOS or Linux

1. Open Terminal.
2. Move into the project folder:

```bash
cd path/to/LonCastAI_project
```

3. Create a virtual environment:

```bash
python3 -m venv venv
```

4. Activate it:

```bash
source venv/bin/activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Start the application:

```bash
python run_app.py
```

7. Open the app in a browser:

```text
http://127.0.0.1:8000
```

### Windows

1. Open PowerShell.
2. Move into the project folder:

```powershell
cd path\to\LonCastAI_project
```

3. Create a virtual environment:

```powershell
py -m venv venv
```

4. Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

5. Install dependencies:

```powershell
pip install -r requirements.txt
```

6. Start the application:

```powershell
python run_app.py
```

7. Open the app in a browser:

```text
http://127.0.0.1:8000
```

### One-click launchers

If preferred, the marker can also use:

- `start_mac.command` on macOS
- `start_windows.bat` on Windows

These scripts create a virtual environment if needed, install dependencies, and start the app.

## How to use the system

1. Open the homepage in the browser.
2. Type a postcode into the postcode field.
3. Pick a matching postcode suggestion if shown.
4. Choose the correct property from the property suggestions list.
5. Press `Search`.
6. Review the prediction summary, map, explanation panel, confidence indicator, and local context.
7. Optionally use the contact section to send a message.

## Test credentials or sample inputs

No login is required.

Sample postcodes that are loaded in the current project dataset:

- `SW5 9SX`
- `SW7 1RH`
- `W8 6SU`
- `SW3 6SH`

These are useful for testing because the current processed dataset is intentionally limited to selected postcode areas.

## Optional configuration

### `.env` file

The application can run without a `.env` file, but some optional features use it.

Create a local `.env` file only if you want to enable these extras:

- Ordnance Survey map tiles
- backend contact form email sending

Use `.env.example` as the template.

### OS Maps API

If you want the app to use Ordnance Survey map tiles, put your key in `.env`:

```text
OS_MAPS_API_KEY=your_key_here
OS_MAPS_LAYER=Light_3857
```

If this key is not provided, the app still runs and uses the fallback map.

### Contact form email

If you want the contact form to send real email from the backend, configure SMTP in `.env`:

```text
CONTACT_TO_EMAIL=aderk001@campus.goldsmiths.ac.uk
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@example.com
SMTP_PASSWORD=your_app_password
SMTP_FROM_EMAIL=your_email@example.com
SMTP_USE_TLS=true
```

Without SMTP configuration, the rest of the app still works, but the contact form will show a clear error instead of sending email.

## Known limitations / not implemented yet

- The current processed project scope is limited to selected postcode areas: `W8`, `SW7`, `SW5`, and `SW3`.
- The application does not currently cover all London postcodes unless the data pipeline is rebuilt with a larger scope.
- Prediction quality depends on available matched sold-price data, so some properties are explained using stronger local context than others.
- The original brief proposed broader dashboard-style insight features and postcode comparison, but the final system keeps the main focus on property-level prediction and explanation.
- The final implementation is designed for local execution and marking; it has not been deployed to a public production server in this submission.
- The final implementation uses FastAPI, a static frontend, and SQLite for portability, rather than the React, PostgreSQL, and cloud deployment approach discussed in the original design brief.
- The contact form only sends real email if SMTP has been configured locally.
- Raw dataset rebuild is not part of the normal marker workflow and may take significantly longer on another machine.

## Optional developer tasks

These are not required for marking, but are included for completeness.

### Rebuild processed data

If raw data has been updated and you need to rebuild the processed property features:

```bash
export PYTHONPATH=.
export REFRESH_PROCESSED_DATASETS=1

python scripts/load_london_postcodes.py
python scripts/load_epc_properties.py
python scripts/load_address_lookup.py

unset REFRESH_PROCESSED_DATASETS

python scripts/build_property_features.py
python scripts/enrich_postcode_price_stats.py
python scripts/enrich_location_features.py
python scripts/enrich_crime_features.py
python scripts/enrich_hpi_features.py
```

### Retrain the ML models

```bash
export PYTHONPATH=.
python scripts/train_property_ml_models.py
```

Laptop-friendly example:

```bash
export PYTHONPATH=.
ML_CV_FOLDS=3 ML_RF_MAX_ESTIMATORS=70 ML_RF_STEP_ESTIMATORS=10 ML_RF_GRID_PATIENCE=5 python scripts/train_property_ml_models.py
```

## Submission note

For a GitHub submission, do not commit the real `.env` file. Keep secrets local and only include `.env.example` with placeholder values.
