"""
Deploy dashboard-specific BigQuery views for Looker Studio.

These views are optimized for visualization with:
- Pre-computed categories and labels
- Weekly/monthly aggregations
- Rolling averages and trends
- Correlation-ready lagged metrics

Usage:
    python deploy_dashboard_views.py
"""

import os
from google.cloud import bigquery

CREDENTIALS_PATH = "/Users/tolgasabanoglu/Desktop/github/spatiotemporal/spatiotemporal-key.json"
SQL_FILE = os.path.join(os.path.dirname(__file__), "looker_views.sql")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH


def deploy_views():
    """Deploy dashboard views to BigQuery."""
    client = bigquery.Client()

    with open(SQL_FILE, "r") as f:
        sql_content = f.read()

    # Split into individual statements
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

    statements = [s for s in statements if 'CREATE OR REPLACE VIEW' in s]

    print(f"Deploying {len(statements)} dashboard views...\n")

    for i, sql in enumerate(statements, 1):
        view_name = sql.split('`')[1] if '`' in sql else f"view_{i}"
        print(f"[{i}/{len(statements)}] {view_name}...", end=" ")

        try:
            job = client.query(sql)
            job.result()
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")

    print("\n" + "=" * 50)
    print("Dashboard views deployed!")
    print("=" * 50)
    print("\nAvailable views:")
    print("  - garmin_data.v_dashboard_daily")
    print("  - garmin_data.v_dashboard_weekly")
    print("  - garmin_data.v_dashboard_monthly")
    print("  - garmin_data.v_dashboard_correlations")
    print("  - garmin_data.v_dashboard_trends")
    print("\nConnect these to Looker Studio via BigQuery connector.")


if __name__ == "__main__":
    deploy_views()
