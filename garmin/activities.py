import os
import json
import time
from garminconnect import Garmin
from dotenv import load_dotenv

# ---- Load environment variables ----
load_dotenv()
EMAIL = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")

if not EMAIL or not PASSWORD:
    raise ValueError("Missing USERNAME or PASSWORD environment variables")

# ---- Setup directories ----
RAW_DIR = "/Users/tolgasabanoglu/Desktop/github/spatiotemporal/data/raw"
ACTIVITIES_DIR = os.path.join(RAW_DIR, "activities")
os.makedirs(ACTIVITIES_DIR, exist_ok=True)

# ---- Login to Garmin ----
print(" Logging into Garmin Connect...")
client = Garmin(EMAIL, PASSWORD)
client.login()

# ---- Helper: Save JSON data ----
def save_json(data, filename):
    path = os.path.join(ACTIVITIES_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f" Saved → {filename}")

# ---- Fetch Activities List ----
print("\n Fetching activities list...")
activities_list_file = "activities_list.json"
activities_list_path = os.path.join(ACTIVITIES_DIR, activities_list_file)

# Fetch activities (adjust limit as needed - max is typically 100 per call)
LIMIT = 100  # Get last 100 activities
activities = []

if os.path.exists(activities_list_path):
    print(f" Loading existing activities list from {activities_list_file}")
    with open(activities_list_path, "r") as f:
        activities = json.load(f)
else:
    try:
        activities = client.get_activities(0, LIMIT)
        if activities:
            save_json(activities, activities_list_file)
            print(f" Found {len(activities)} activities")
        else:
            print(" No activities found")
            exit()
    except Exception as e:
        print(f" Error fetching activities list: {e}")
        exit()

# ---- Fetch Detailed Data for Each Activity ----
print(f"\n Fetching detailed data for {len(activities)} activities...\n")

for i, activity in enumerate(activities, 1):
    activity_id = activity.get('activityId')
    activity_name = activity.get('activityName', 'Unnamed')
    activity_type = activity.get('activityType', {}).get('typeKey', 'unknown')
    start_time = activity.get('startTimeLocal', 'unknown')
    
    print(f"[{i}/{len(activities)}] {activity_type}: {activity_name} ({start_time})")
    
    # Skip if already downloaded
    detail_file = f"activity_{activity_id}_detail.json"
    detail_path = os.path.join(ACTIVITIES_DIR, detail_file)
    
    if os.path.exists(detail_path):
        print(f"  ⏭ Already exists, skipping")
        continue
    
    try:
        # Fetch detailed activity data
        details = client.get_activity(activity_id)
        
        if details:
            save_json(details, detail_file)
        else:
            print(f"   No detail data available")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    # Rate limiting - be nice to Garmin's servers
    time.sleep(1)

print("\n Done! All activities downloaded.")
print(f" Files saved to: {ACTIVITIES_DIR}")