import os
import json
import datetime
import time
import yaml
from garminconnect import Garmin

# ---- Load credentials from config.yaml ----
with open("/Users/tolgasabanoglu/Desktop/github/spatiotemporal/config/config.yaml") as f:
    creds = yaml.safe_load(f)

EMAIL = creds["email"]
PASSWORD = creds["password"]

# ---- Setup raw data directory ----
raw_dir = "../data/raw/"
os.makedirs(raw_dir, exist_ok=True)

# ---- Login to Garmin ----
print("🔐 Logging into Garmin Connect...")
client = Garmin(EMAIL, PASSWORD)
client.login()

# ---- Save JSON data if valid ----
def save_json(data, name, date_str):
    filename = f"{name}_{date_str}.json"
    path = os.path.join(raw_dir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {name} → {filename}")

# ---- Garmin metric fetchers ----
fetchers = {
    "steps": client.get_steps_data,
    "sleep": client.get_sleep_data,
    "stress": client.get_stress_data,
    "body_battery": client.get_body_battery,
    "heart_rate": client.get_heart_rates,
}

# ---- Set known start date ----
start_date = datetime.date(2025, 2, 25)
today = datetime.date.today()

print(f"\n📆 Fetching Garmin data from {start_date} to {today}\n")

# ---- Loop over dates and metrics ----
for single_date in (start_date + datetime.timedelta(n) for n in range((today - start_date).days + 1)):
    date_str = single_date.isoformat()
    print(f"\n📅 {date_str}")

    for name, func in fetchers.items():
        filename = f"{name}_{date_str}.json"
        file_path = os.path.join(raw_dir, filename)

        if os.path.exists(file_path):
            print(f"⏭️  Skipping {filename} (already exists)")
            continue

        try:
            print(f"📦 Fetching {name}...")
            data = func(date_str)

            # Check if the data is valid (non-empty)
            is_valid = False
            if isinstance(data, list):
                is_valid = len(data) > 0
            elif isinstance(data, dict):
                is_valid = any(
                    v not in (None, 0, [], {}, "", "null") for v in data.values()
                )

            if is_valid:
                save_json(data, name, date_str)
            else:
                print(f"🚫 No valid {name} data for {date_str}, skipping.")

        except Exception as e:
            print(f"⚠️ Error fetching {name} on {date_str}: {e}")

    time.sleep(1)  # Delay to avoid rate limits
