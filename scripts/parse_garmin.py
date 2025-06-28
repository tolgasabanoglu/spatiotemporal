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

# ---- Setup ----
raw_dir = "../data/raw/"
os.makedirs(raw_dir, exist_ok=True)

# ---- Login ----
print("🔐 Logging into Garmin Connect...")
client = Garmin(EMAIL, PASSWORD)
client.login()

# ---- Save helper ----
def save_json(data, name, date_str):
    filename = f"{name}_{date_str}.json"
    path = os.path.join(raw_dir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {name} to {path}")

# ---- Fetchers ----
fetchers = {
    "steps": client.get_steps_data,
    "sleep": client.get_sleep_data,
    "stress": client.get_stress_data,
    "body_battery": client.get_body_battery,
    "heart_rate": client.get_heart_rates,
    "respiration": client.get_respiration_data,
    "activities": client.get_activities,
}

# ---- Date range: last 30 days ----
today = datetime.date.today()
start_date = today - datetime.timedelta(days=30)

print(f"\n📆 Fetching Garmin data from {start_date} to {today}\n")

for day_offset in range(31):
    date = start_date + datetime.timedelta(days=day_offset)
    date_str = date.isoformat()
    print(f"\n📅 Date: {date_str}")

    for name, func in fetchers.items():
        try:
            print(f"📦 Fetching {name}...")
            data = func(date_str)
            save_json(data, name, date_str)
        except Exception as e:
            print(f"⚠️ Failed to fetch {name} for {date_str}: {e}")

    time.sleep(1)  # Delay to avoid rate limits
