"""
Adds the 8 widgets from dashboards/2026_live_season.sql to the "2026
Season" page of the existing Lakeview dashboard, via the Databricks REST
API -- an alternative to manually pasting each query into the SQL Editor
one at a time.

NOT independently tested against a real Databricks workspace (no network
access to one from where this was written) -- built from the exact JSON
shapes confirmed working in this workspace's own exported
2025_season_recap.lvdash.json, so it should be low-risk, but treat the
first run as an experiment, not a sure thing.

Safety:
  - Defaults to a DRY RUN: fetches the dashboard, shows what it would add,
    and writes nothing back unless you pass --confirm.
  - Always backs up the dashboard's current serialized_dashboard JSON to a
    local timestamped file before attempting any write, so you can restore
    it by hand (paste the backed-up JSON back via PATCH, or re-import it)
    if a write goes wrong.

Setup (PowerShell), before running:
    $env:DATABRICKS_HOST = "https://dbc-7fa5604e-6f98.cloud.databricks.com"
    $env:DATABRICKS_TOKEN = "dapi...your-token..."

Usage:
    python dashboards/add_2026_queries_via_api.py                # dry run
    python dashboards/add_2026_queries_via_api.py --confirm       # writes for real
"""

import json
import os
import re
import secrets
import sys
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError

DASHBOARD_ID = "01f1adecbbb61856907695eab9c5db41"
TARGET_PAGE_NAME = "2026 Season"
SQL_FILE = os.path.join(os.path.dirname(__file__), "2026_live_season.sql")


def api_request(method, path, host, token, body=None):
    url = f"{host}/api/2.0{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        print(f"HTTP {e.code} calling {method} {path}")
        print(e.read().decode("utf-8"))
        raise


def parse_sql_file(path):
    """Returns an ordered list of (title, sql_text) using this repo's
    "-- Widget N: <title> ..." comment headers, ignoring the header block
    before Widget 1. Robust to semicolons appearing inside comment text
    (an earlier naive split-on-';' approach broke on that)."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    widgets = []
    current_title = None
    current_sql_lines = []

    header_re = re.compile(r"^--\s*Widget\s+\d+[a-z]?:\s*(.+?)\s*(\(.*)?$")

    def flush():
        if current_title is not None:
            sql = "\n".join(current_sql_lines).strip()
            if sql:
                widgets.append((current_title, sql))

    for line in lines:
        m = header_re.match(line.strip())
        if m:
            flush()
            current_title = m.group(1).strip()
            current_sql_lines = []
            continue
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        if current_title is not None:
            current_sql_lines.append(line.rstrip("\n"))
    flush()
    return widgets


def table_widget(name, dataset_name, columns, title):
    return {
        "widget": {
            "name": name,
            "queries": [
                {
                    "name": "main_query",
                    "query": {
                        "datasetName": dataset_name,
                        "fields": [{"name": c, "expression": f"`{c}`"} for c in columns],
                        "disaggregated": True,
                    },
                }
            ],
            "spec": {
                "version": 2,
                "widgetType": "table",
                "encodings": {"columns": [{"fieldName": c} for c in columns]},
                "frame": {"title": title, "showTitle": True},
                "data": {"queryName": "main_query"},
            },
        }
    }


def bar_widget(name, dataset_name, x_field, y_field, title):
    return {
        "widget": {
            "name": name,
            "queries": [
                {
                    "name": "main_query",
                    "query": {
                        "datasetName": dataset_name,
                        "fields": [
                            {"name": x_field, "expression": f"`{x_field}`"},
                            {"name": y_field, "expression": f"`{y_field}`"},
                        ],
                        "disaggregated": True,
                    },
                }
            ],
            "spec": {
                "version": 3,
                "widgetType": "bar",
                "encodings": {
                    "x": {"fieldName": x_field, "scale": {"type": "categorical"}, "displayName": x_field.title()},
                    "y": {"fieldName": y_field, "scale": {"type": "quantitative"}, "displayName": y_field.title()},
                },
                "frame": {"title": title, "showTitle": True},
            },
        }
    }


def line_widget(name, dataset_name, x_field, y_field, color_field, title, y_display_name=None):
    fields = [
        {"name": x_field, "expression": f"`{x_field}`"},
        {"name": y_field, "expression": f"`{y_field}`"},
    ]
    encodings = {
        "x": {"fieldName": x_field, "scale": {"type": "quantitative"}, "displayName": x_field.title()},
        "y": {"fieldName": y_field, "scale": {"type": "quantitative"}, "displayName": y_display_name or y_field.title()},
    }
    if color_field:
        fields.append({"name": color_field, "expression": f"`{color_field}`"})
        encodings["color"] = {"fieldName": color_field, "scale": {"type": "categorical"}, "displayName": color_field.title()}
    return {
        "widget": {
            "name": name,
            "queries": [
                {
                    "name": "main_query",
                    "query": {"datasetName": dataset_name, "fields": fields, "disaggregated": True},
                }
            ],
            "spec": {
                "version": 3,
                "widgetType": "line",
                "encodings": encodings,
                "frame": {"title": title, "showTitle": True},
            },
        }
    }


def counter_widget(name, dataset_name, field, title, as_percent=False):
    value_encoding = {"fieldName": field}
    if as_percent:
        value_encoding["format"] = {"type": "number-percent", "decimalPlaces": {"type": "max", "places": 2}}
    return {
        "widget": {
            "name": name,
            "queries": [
                {
                    "name": "main_query",
                    "query": {
                        "datasetName": dataset_name,
                        "fields": [{"name": field, "expression": f"`{field}`"}],
                        "disaggregated": True,
                    },
                }
            ],
            "spec": {
                "version": 2,
                "widgetType": "counter",
                "encodings": {"value": value_encoding},
                "frame": {"title": title, "showTitle": True},
                "data": {"queryName": "main_query"},
            },
        }
    }


def main():
    confirm = "--confirm" in sys.argv
    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    if not host or not token:
        print("Set DATABRICKS_HOST and DATABRICKS_TOKEN first (see script docstring).")
        sys.exit(1)
    host = host.rstrip("/")

    print(f"Fetching dashboard {DASHBOARD_ID} ...")
    dash = api_request("GET", f"/lakeview/dashboards/{DASHBOARD_ID}", host, token)

    backup_path = f"dashboard_backup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(dash, f, indent=2)
    print(f"Backed up current dashboard to {backup_path}")

    definition = json.loads(dash["serialized_dashboard"])

    queries = parse_sql_file(SQL_FILE)
    titles = [t for t, _ in queries]
    print(f"Parsed {len(queries)} queries from {SQL_FILE}: {titles}")

    def sql_for(title_substr):
        for t, sql in queries:
            if title_substr.lower() in t.lower():
                return sql
        raise KeyError(f"No query found matching '{title_substr}' -- check dashboards/2026_live_season.sql headers")

    new_datasets = [
        {"name": "ds2026_standings", "displayName": "Final Standings 2026", "queryLines": [sql_for("Final Standings")]},
        {"name": "ds2026_wins_by_team", "displayName": "Wins by Team 2026", "queryLines": [sql_for("Wins by Team")]},
        {"name": "ds2026_weekly_trend", "displayName": "Weekly Points Trend 2026", "queryLines": [sql_for("Weekly Points Trend")]},
        {"name": "ds2026_accuracy_counter", "displayName": "Matchup Prediction Accuracy 2026", "queryLines": [sql_for("Matchup Prediction Accuracy")]},
        {"name": "ds2026_accuracy_by_week", "displayName": "Accuracy by Week 2026", "queryLines": [sql_for("Prediction Accuracy by Week")]},
        {"name": "ds2026_matchup_detail", "displayName": "Predicted vs Actual Matchup Detail 2026", "queryLines": [sql_for("Matchup Detail")]},
        {"name": "ds2026_championship_tracker", "displayName": "Championship Tracker 2026", "queryLines": [sql_for("Championship Tracker")]},
        {"name": "ds2026_bracket", "displayName": "Playoff Bracket 2026", "queryLines": [sql_for("Playoff Bracket")]},
    ]

    def wid():
        return secrets.token_hex(4)

    new_widgets = [
        (table_widget(wid(), "ds2026_standings",
            ["team", "wins", "losses", "ties", "points_for", "points_against", "point_diff"],
            "Final Standings"), 4, 6),
        (bar_widget(wid(), "ds2026_wins_by_team", "team", "wins", "Wins by Team"), 4, 6),
        (line_widget(wid(), "ds2026_weekly_trend", "week", "points_for", "team",
            "Weekly Points Trend", y_display_name="Points"), 12, 6),
        (counter_widget(wid(), "ds2026_accuracy_counter", "accuracy_pct",
            "Matchup Prediction Accuracy", as_percent=False), 4, 3),
        (bar_widget(wid(), "ds2026_accuracy_by_week", "week", "accuracy_pct", "Accuracy by Week"), 4, 4),
        (table_widget(wid(), "ds2026_matchup_detail",
            ["team", "week", "opponent", "predicted_points_for", "actual_points_for",
             "predicted_result", "actual_result", "correct"],
            "Predicted vs Actual Matchup Detail"), 12, 6),
        (table_widget(wid(), "ds2026_championship_tracker",
            ["as_of_week", "seed1", "seed2", "seed3", "seed4", "seed5", "seed6", "predicted_champion"],
            "Championship Tracker"), 8, 6),
        (table_widget(wid(), "ds2026_bracket",
            ["scenario", "round", "week", "team_a", "score_a", "team_b", "score_b", "winner"],
            "Playoff Bracket: Predicted vs Actual"), 12, 8),
    ]

    definition.setdefault("datasets", [])
    existing_ds_names = {d["name"] for d in definition["datasets"]}
    for ds in new_datasets:
        if ds["name"] not in existing_ds_names:
            definition["datasets"].append(ds)

    target_page = None
    for page in definition.get("pages", []):
        if page.get("displayName") == TARGET_PAGE_NAME:
            target_page = page
            break

    if target_page is None:
        print(f'No page named "{TARGET_PAGE_NAME}" found -- creating it.')
        target_page = {
            "name": secrets.token_hex(4),
            "displayName": TARGET_PAGE_NAME,
            "layout": [],
            "pageType": "PAGE_TYPE_CANVAS",
            "layoutVersion": "GRID_V1",
        }
        definition.setdefault("pages", []).insert(-1, target_page) if definition.get("pages") and definition["pages"][-1].get("pageType") == "PAGE_TYPE_GLOBAL_FILTERS" else definition.setdefault("pages", []).append(target_page)

    layout = target_page.setdefault("layout", [])
    y_cursor = max([item["position"]["y"] + item["position"]["height"] for item in layout], default=0)
    x_cursor = 0
    row_height = 0

    for widget_def, width, height in new_widgets:
        if x_cursor + width > 12:
            x_cursor = 0
            y_cursor += row_height
            row_height = 0
        entry = dict(widget_def)
        entry["position"] = {"x": x_cursor, "y": y_cursor, "width": width, "height": height}
        layout.append(entry)
        x_cursor += width
        row_height = max(row_height, height)

    new_serialized = json.dumps(definition)

    print()
    print(f"Would add {len(new_datasets)} datasets and {len(new_widgets)} widgets to page '{TARGET_PAGE_NAME}'.")
    print(f"New total size of serialized_dashboard: {len(new_serialized)} chars (was {len(dash['serialized_dashboard'])}).")

    if not confirm:
        preview_path = "dashboard_update_preview.json"
        with open(preview_path, "w", encoding="utf-8") as f:
            json.dump(definition, f, indent=2)
        print(f"\nDRY RUN -- nothing written. Full proposed dashboard saved to {preview_path} for review.")
        print("Re-run with --confirm to actually PATCH the dashboard.")
        return

    patch_body = {
        "display_name": dash.get("display_name"),
        "serialized_dashboard": new_serialized,
    }
    if dash.get("warehouse_id"):
        patch_body["warehouse_id"] = dash["warehouse_id"]

    print("\nSending PATCH to update the dashboard ...")
    result = api_request("PATCH", f"/lakeview/dashboards/{DASHBOARD_ID}", host, token, body=patch_body)
    print("Success. Response:")
    print(json.dumps(result, indent=2)[:1000])


if __name__ == "__main__":
    main()
