import os
import json
from google.cloud import bigquery

# ------------------- Configuration -------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREDENTIALS_PATH = os.path.join(PROJECT_ROOT, "spatiotemporal-key.json")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATASET_NAME = "garmin_data"
TABLE_NAME = "garmin_raw_data"
CHUNK_SIZE = 200  # rows per batch

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH

# ------------------- Initialize BigQuery client -------------------
client = bigquery.Client()

# ------------------- Ensure dataset exists -------------------
dataset_id = f"{client.project}.{DATASET_NAME}"
try:
    client.get_dataset(dataset_id)
    print(f" Dataset exists: {dataset_id}")
except Exception:
    dataset = bigquery.Dataset(dataset_id)
    client.create_dataset(dataset)
    print(f" Created dataset: {dataset_id}")

# ------------------- Prepare rows from JSON files -------------------
def prepare_rows(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    rows = []
    for file in files:
        path = os.path.join(directory, file)
        try:
            with open(path, "r") as f:
                data = json.load(f)
                # store as JSON string in raw_json field
                rows.append({"filename": file, "raw_json": json.dumps(data)})
        except Exception as e:
            print(f" Failed to read {file}: {e}")
    print(f" Prepared {len(rows)} rows for upload")
    return rows

# ------------------- Create or reset table -------------------
def create_table_if_needed(table_id, reset=False):
    try:
        table = client.get_table(table_id)
        if reset:
            print(f" Resetting table: {table_id}")
            client.delete_table(table_id)
            raise Exception("Table reset requested")
        # check schema, add missing fields if needed
        required_fields = {"filename", "raw_json"}
        existing_fields = {field.name for field in table.schema}
        if not required_fields.issubset(existing_fields):
            print(f" Schema mismatch detected. Dropping and recreating table...")
            client.delete_table(table_id)
            raise Exception("Table recreated due to schema mismatch")
        print(f" Table exists: {table_id}")
    except Exception:
        schema = [
            bigquery.SchemaField("filename", "STRING"),
            bigquery.SchemaField("raw_json", "STRING")
        ]
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)
        print(f" Created table with schema: {table_id}")

# ------------------- Upload rows in chunks -------------------
def upload_rows_to_bigquery(table_id, rows, chunk_size=CHUNK_SIZE):
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        print(f" Uploading chunk {i // chunk_size + 1}/{(len(rows) - 1) // chunk_size + 1}...")
        job = client.insert_rows_json(table_id, chunk)
        if job:
            print(f" Errors occurred: {job}")
        else:
            print(f" Uploaded {len(chunk)} rows successfully")

# ------------------- Main -------------------
def main(reset_table=True):
    """
    Upload raw JSON data to BigQuery.
    Args:
        reset_table: If True, drops and recreates the table to avoid duplicates.
    """
    table_id = f"{dataset_id}.{TABLE_NAME}"
    create_table_if_needed(table_id, reset=reset_table)
    rows = prepare_rows(RAW_DIR)
    upload_rows_to_bigquery(table_id, rows)

if __name__ == "__main__":
    print(" Starting Garmin → BigQuery Raw Upload")
    main()
