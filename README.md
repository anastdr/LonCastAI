# LonCastAI

## User guide

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

This submission follows the brief by prioritising the core predictive function over secondary features and main implemented focus is:

- short-term property price estimation
- explainable output rather than black-box prediction only
- minimal user input through postcode search and property selection
- user-facing clarity through readable text, map context, and prediction breakdowns

The original brief also discussed broader dashboard-style insight features, postcode comparison, cloud deployment, and a PostgreSQL-backed architecture. For the final submission, these were simplified where needed to keep the project feasible, reliable, and easy to run on another machine.

## Project structure

- `backend/` FastAPI backend, prediction logic, routes, and database access
- `frontend/` HTML, CSS, JavaScript, branding, map, and search interface
- `db/submission_database.db` compact SQLite database recommended for the submitted version
- `db/database.db` full local development SQLite database used during project work - not uploaded to git due to big size (more than 2GB)
- `models/property_ml_models.pkl` trained ML model artifact used by the app
- `scripts/` data processing and ML training scripts
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

Use the steps below to run the project from the submitted folder without rebuilding the raw data

### Important note

The core application does not require a `.env` file to run.

- If no `.env` file is provided, the main search and prediction workflow still works.
- Without `.env`, the app will use the fallback map.
- Without `.env`, the contact form email feature will not send real emails and will show a configuration error instead.

For GitHub submission the file with database is 

- `db/submission_database.db`

This compact database is suitable for the selected project postcodes and is small enough to include in a normal repository. Application therefore automatically uses `db/submission_database.db` if `db/database.db` is not present

The submitted database is a post-processed project dataset created by combining multiple raw data sources, cleaning inconsistent records, matching property information across datasets, and enriching the final data with features such as sold-price history, school and station proximity, crime context, and house price index values. Unfortunately, the initial database size was too large (more than 1.5 GB) which is not the size allowed for GitHub thus the solution above was implemented 

### Quick start

If you download the project from GitHub, download it as a ZIP, extract it, and run the commands below from inside the extracted project folder

### Before starting

Make sure the submitted project folder includes these files:

- `db/submission_database.db` or `db/database.db`
- `models/property_ml_models.pkl`
- `frontend/`
- `backend/`
- `requirements.txt`
- `run_app.py`

If both database files are missing, or if `models/property_ml_models.pkl` is missing, the application can still open, but property search or ML predictions will not work correctly.

The submitted folder should include:

- `backend/`
- `frontend/`
- `scripts/`
- `db/submission_database.db`
- `models/property_ml_models.pkl`
- `README.md`
- `requirements.txt`
- `run_app.py`
- `start_mac.command`
- `start_windows.bat`
- `.env.example`

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

8. Test the app using one of the sample postcodes from the section below.

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

8. Test the app using one of the sample postcodes from the section below.

### One-click launchers

If preferred, you can also use:

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

SW7 3EG
SW7 2ST
W8 5PT
SW3 1AW

These are useful for testing because the current processed dataset is intentionally limited to selected postcode areas.

Test flow:

1. Enter one of the sample postcodes above.
2. Choose a property from the suggestion list.
3. Press `Search`.
4. Check that a prediction summary, explanation text, and map section are displayed.

## Optional configuration

### `.env` file

The application can run without a `.env` file, but some optional features use it

Create a local `.env` file if you want to enable these extras:

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

The contact form sends email from the backend server only if SMTP environment variables are configured in a local `.env` file.

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

Without SMTP configuration, the rest of the app still works, but the contact form will show a clear error instead of sending email. This is expected behaviour and does not affect the main prediction functionality.

## Known limitations / not implemented yet

- The current processed project scope is limited to selected postcode areas: `W8`, `SW7`, `SW5`, and `SW3`.
- The application does not currently cover all London postcodes unless the data pipeline is rebuilt with a larger scope due to computational limitation of local machince and resource lack to use external/cloud services.
- Prediction quality depends on available matched sold-price data, so some properties are explained using stronger local context than others.
- Only properties that exist in the processed submission database can be searched successfully.
- Some valid London postcodes will therefore return no matching property data in this submitted version.
- Original brief proposed broader dashboard-style insight features and postcode comparison but the final system keeps the main focus on property-level prediction and explanation due to the computational limitations.
- Final implementation is designed for local execution it has not been deployed to a public production server in this submission
- App uses FastAPI, a static frontend, and SQLite for portability, rather than the React, PostgreSQL, and cloud deployment approach discussed in the original design brief
- The contact form only sends real email if SMTP has been configured locally.
- The app uses a compact `submission_database.db` for the GitHub-ready version, so the submitted version is smaller in scope than the full local development database
- The map experience depends partly on internet access because online map tiles are loaded in the browser.
- Raw dataset rebuild is not part of the normal run instructions and may take significantly longer on another machine

During the project development, the main focus shifted towards processing and structuring suitable datasets to support a reliable machine-learning pipeline and property price predictions, with particular emphasis on qualitative features such as proximity to schools and stations, recent crime levels, and the House Price Index. These features were informed by earlier stakeholder interviews with estate agents in order to align the final system with the main project objectives






## Datasets used and considered

TFL dataset -  https://tfl.gov.uk/info-for/open-data-users/our-open-data?utm_source=chatgpt.com#on-this-page-1

Price paid dataset : https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads?utm_source=chatgpt.com#february-2026-data-current-month

Station footfall - https://tfl.gov.uk/corporate/publications-and-reports/network-demand-data?utm_source=chatgpt.com#on-this-page-2

School rating - https://www.gov.uk/government/statistical-data-sets/monthly-management-information-ofsteds-school-inspections-outcomes?utm_source=chatgpt.com

School proximity - https://get-information-schools.service.gov.uk/Downloads
Crime levels - https://data.london.gov.uk/dataset/mps-recorded-crime-geographic-breakdown-exy3m/

Deprivation areas - https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019?utm_source=chatgpt.com

House price index - https://www.gov.uk/government/collections/uk-house-price-index-reports?utm_source=chatgpt.com

Greenery proximity - https://osdatahub.os.uk/data/downloads/open

EPC ratings - https://epc.opendatacommunities.org/docs/api

MPC recorded crime : https://data.london.gov.uk/dataset/mps-recorded-crime-geographic-breakdown-exy3m/
