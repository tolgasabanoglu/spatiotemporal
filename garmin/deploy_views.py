"""
Deploy BigQuery views for Garmin data transformation.

This script creates/updates views that:
- Parse raw JSON data into structured tables
- Handle data quality (NULL for missing/invalid data like 0 steps)
- Merge all metrics into a single daily view

Usage:
    python deploy_views.py
"""

import os
from google.cloud import bigquery

# Configuration
CREDENTIALS_PATH = "/Users/tolgasabanoglu/Desktop/github/spatiotemporal/spatiotemporal-key.json"
SQL_FILE = os.path.join(os.path.dirname(__file__), "sql", "views.sql")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH


def deploy_views():
    """Deploy all views from the SQL file."""
    client = bigquery.Client()

    # Read SQL file
    with open(SQL_FILE, "r") as f:
        sql_content = f.read()

    # Split into individual statements (by CREATE OR REPLACE VIEW)
    statements = []
    current = []

    for line in sql_content.split('\n'):
        if line.strip().startswith('CREATE OR REPLACE VIEW') and current:
            statements.append('\n'.join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        statements.append('\n'.join(current))

    # Filter out empty/comment-only statements
    statements = [s for s in statements if 'CREATE OR REPLACE VIEW' in s]

    print(f"Found {len(statements)} views to deploy\n")

    # Execute each statement
    for i, sql in enumerate(statements, 1):
        # Extract view name
        view_name = sql.split('`')[1] if '`' in sql else f"view_{i}"

        print(f"[{i}/{len(statements)}] Deploying {view_name}...")

        try:
            job = client.query(sql)
            job.result()  # Wait for completion
            print(f"    ✅ Success")
        except Exception as e:
            print(f"    ❌ Error: {e}")

    print("\n" + "=" * 50)
    print("Deployment complete!")
    print("=" * 50)

    # Show data quality summary
    print("\nData Quality Summary:")
    print("-" * 50)

    try:
        query = """
        SELECT * FROM `garmin_data.v_data_quality_summary`
        ORDER BY month
        """
        df = client.query(query).to_dataframe()
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Could not fetch summary: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("Garmin BigQuery Views Deployment")
    print("=" * 50 + "\n")
    deploy_views()
