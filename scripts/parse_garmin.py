import subprocess
import os
import json
import datetime
from google.cloud import bigquery

# ---- Step 1: Run parse_garmin.py to get latest Garmin data ----
print("🔄 Running parse_garmin.py to fetch latest Garmin data...")
subprocess.run(["python", "scripts/parse_garmin.py"], check=True)

# ---- Step 2: Setup BigQuery client and dataset ----
PROJECT_ID = os.environ.get("PROJECT_ID")  # make sure you exported it
DATA_DIR = "/Users/tolgasabanoglu/Desktop/github/spatiotemporal/data/raw"
DATASET_NAME = "garmin_data"

client = bigquery.Client(project=PROJECT_ID)

# Create dataset if it doesn't exist
dataset_ref = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_NAME}")
try:
    client.get_dataset(dataset_ref)
    print(f"Dataset {DATASET_NAME} exists ✅")
except Exception:
    print(f"Creating dataset {DATASET_NAME}...")
    client.create_dataset(dataset_ref)
    print(f"Dataset {DATASET_NAME} created ✅")

# ---- Helper to create table if it doesn't exist ----
def create_table_if_not_exists(table_id, sample_row):
    try:
        client.get_table(table_id)
        print(f"Table {table_id} exists ✅")
    except Exception:
        print(f"📦 Creating table {table_id}...")
        schema = []
        for key, value in sample_row.items():
            if isinstance(value, int):
                field_type = "INTEGER"
            elif isinstance(value, float):
                field_type = "FLOAT"
            elif isinstance(value, str):
                field_type = "STRING"
            else:
                field_type = "STRING"
            schema.append(bigquery.SchemaField(key, field_type))
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)
        print(f"✅ Table {table_id} created")

# ---- Load JSON files grouped by day ----
def load_json_files(directory, days_back=90):
    rows_by_day = {}
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=days_back)

    for file in os.listdir(directory):
        if not file.endswith(".json"):
            continue
        path = os.path.join(directory, file)
        with open(path) as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load {file}: {e}")
                continue

        # Handle both list and dict JSONs
        if isinstance(data, dict):
            entries = [data]
        elif isinstance(data, list):
            entries = data
        else:
            continue

        # Extract date from filename (assumes format name_YYYY-MM-DD.json)
        try:
            date_part = file.split("_")[-1].replace(".json", "")
            file_date = datetime.date.fromisoformat(date_part)
        except Exception as e:
            print(f"⚠️ Could not parse date from {file}: {e}")
            continue

        if file_date < start_date:
            continue

        rows_by_day.setdefault(file_date.isoformat(), []).extend(entries)

    return rows_by_day

# ---- Upload rows to BigQuery in daily tables ----
def upload_rows(dataset_name, rows_by_day):
    for day, rows in rows_by_day.items():
        table_name = f"garmin_{day.replace('-', '_')}"
        table_id = f"{PROJECT_ID}.{dataset_name}.{table_name}"

        if not rows:
            continue

        create_table_if_not_exists(table_id, rows[0])

        # Insert rows in batches of 100
        batch_size = 100
        print(f"⚡ Uploading {len(rows)} rows to {table_name} in batches of {batch_size}...")
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            errors = client.insert_rows_json(table_id, batch)
            if errors:
                print(f"❌ Errors inserting batch {i}-{i + batch_size}: {errors}")

# ---- Main ----
def main():
    rows_by_day = load_json_files(DATA_DIR, days_back=90)
    if not rows_by_day:
        print("🚫 No JSON rows found to upload")
        return
    upload_rows(DATASET_NAME, rows_by_day)
    print("✅ All data uploaded successfully!")

if __name__ == "__main__":
    main()
