import os
import json
import datetime
import time
from garminconnect import Garmin
from dotenv import load_dotenv  # pip install python-dotenv

# ---- Load environment variables ----
load_dotenv()
EMAIL = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

if not EMAIL or not PASSWORD:
    raise ValueError("Missing USERNAME or PASSWORD environment variables")

# ---- Setup raw data directory ----
RAW_DIR = "/Users/tolgasabanoglu/Desktop/github/spatiotemporal/data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# ---- Login to Garmin ----
print("🔐 Logging into Garmin Connect...")
client = Garmin(EMAIL, PASSWORD)
client.login()

# ---- Helper: Save JSON data ----
def save_json(data, name, date_str):
    filename = f"{name}_{date_str}.json"
    path = os.path.join(RAW_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {name} → {filename}")

# ---- Garmin metric fetchers ----
FETCHERS = {
    "steps": client.get_steps_data,
    "sleep": client.get_sleep_data,
    "stress": client.get_stress_data,
    "body_battery": client.get_body_battery,
    "heart_rate": client.get_heart_rates,
}

# ---- Date range ----
START_DATE = datetime.date(2025, 2, 25)
TODAY = datetime.date.today()
print(f"\n📆 Fetching Garmin data from {START_DATE} to {TODAY}\n")

# ---- Loop over dates and metrics ----
for n in range((TODAY - START_DATE).days + 1):
    single_date = START_DATE + datetime.timedelta(n)
    date_str = single_date.isoformat()
    print(f"\n📅 {date_str}")

    for name, func in FETCHERS.items():
        file_path = os.path.join(RAW_DIR, f"{name}_{date_str}.json")

        if os.path.exists(file_path):
            print(f"⏭️ Skipping {name} for {date_str} (already exists)")
            continue

        try:
            print(f"📦 Fetching {name}...")
            data = func(date_str)

            # Validate data
            is_valid = False
            if isinstance(data, list):
                is_valid = len(data) > 0
            elif isinstance(data, dict):
                is_valid = any(v not in (None, 0, [], {}, "", "null") for v in data.values())

            if is_valid:
                save_json(data, name, date_str)
            else:
                print(f"🚫 No valid {name} data for {date_str}, skipping.")

        except Exception as e:
            print(f"⚠️ Error fetching {name} on {date_str}: {e}")

    # Avoid hitting rate limits
    time.sleep(1)
