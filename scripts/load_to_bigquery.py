import os
import json
import datetime
from google.cloud import bigquery

# ------------------- Configuration -------------------
DATA_DIR = "/Users/tolgasabanoglu/Desktop/github/spatiotemporal/data/raw"
PROJECT_ID = os.getenv("PROJECT_ID", "spatiotemporal-473309")
DATASET_NAME = "garmin_data"

# ------------------- Initialize BigQuery client -------------------
client = bigquery.Client(project=PROJECT_ID)

# ------------------- Create dataset if not exists -------------------
def create_dataset(dataset_name):
    dataset_id = f"{PROJECT_ID}.{dataset_name}"
    try:
        client.get_dataset(dataset_id)
        print(f"Dataset {dataset_name} exists ✅")
    except:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        client.create_dataset(dataset)
        print(f"Dataset {dataset_name} created ✅")

# ------------------- Load JSON files from last N days -------------------
def load_json_files(directory, days_back=30):
    rows_by_day = {}
    cutoff_date = datetime.date.today() - datetime.timedelta(days=days_back)
    
    for file in os.listdir(directory):
        if not file.endswith(".json"):
            continue
        
        path = os.path.join(directory, file)
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to read {file}: {e}")
            continue

        # Extract date from filename (assuming format: metric_YYYY-MM-DD.json)
        try:
            date_str = file.split("_")[-1].replace(".json", "")
            file_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception as e:
            print(f"⚠️ Could not parse date from {file}: {e}")
            continue

        if file_date < cutoff_date:
            continue  # skip older files

        # Initialize daily list
        rows_by_day.setdefault(date_str, [])

        # Flatten dict if data is dict
        if isinstance(data, dict):
            rows_by_day[date_str].append(data)
        elif isinstance(data, list):
            rows_by_day[date_str].extend(data)
        else:
            print(f"⚠️ Unexpected data type in {file}")

    return rows_by_day

# ------------------- Create table if not exists -------------------
def create_table_if_not_exists(table_id, sample_row):
    try:
        client.get_table(table_id)
        print(f"Table {table_id} exists ✅")
        return
    except:
        pass

    # Generate schema from sample row
    schema = []
    for key, value in sample_row.items():
        if isinstance(value, int):
            field_type = "INTEGER"
        elif isinstance(value, float):
            field_type = "FLOAT"
        elif isinstance(value, bool):
            field_type = "BOOLEAN"
        else:
            field_type = "STRING"
        schema.append(bigquery.SchemaField(key, field_type))

    table = bigquery.Table(table_id, schema=schema)
    client.create_table(table)
    print(f"📦 Table {table_id} created")

# ------------------- Upload rows to BigQuery -------------------
def upload_rows(dataset_name, table_name, rows):
    table_id = f"{PROJECT_ID}.{dataset_name}.{table_name}"
    create_table_if_not_exists(table_id, rows[0])
    
    # Batch insert to avoid 413 error
    batch_size = 100
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        errors = client.insert_rows_json(table_id, batch)
        if errors:
            print(f"⚠️ Errors while inserting batch: {errors}")
        else:
            print(f"⚡ Uploaded batch {i//batch_size + 1}/{(len(rows)-1)//batch_size + 1}")

# ------------------- Main -------------------
def main():
    create_dataset(DATASET_NAME)
    rows_by_day = load_json_files(DATA_DIR, days_back=30)

    if not rows_by_day:
        print("No JSON rows found for the last 30 days ❌")
        return

    for date_str, rows in rows_by_day.items():
        table_name = f"garmin_{date_str.replace('-', '_')}"
        print(f"\nUploading {len(rows)} rows to table {table_name}...")
        upload_rows(DATASET_NAME, table_name, rows)

    print("\n✅ All data uploaded successfully!")

if __name__ == "__main__":
    main()
