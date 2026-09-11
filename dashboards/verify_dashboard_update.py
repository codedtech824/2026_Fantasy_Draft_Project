"""
Sanity-checks a dry-run proposal from add_2026_queries_via_api.py against
the backup it took, before you commit to --confirm. Run from the same
folder where dashboard_backup_*.json and dashboard_update_preview.json
were created.

Usage:
    python dashboards/verify_dashboard_update.py
"""

import glob
import json


def main():
    backups = sorted(glob.glob("dashboard_backup_*.json"))
    if not backups:
        print("No dashboard_backup_*.json found in this folder -- run the dry run first.")
        return
    backup_file = backups[-1]

    with open(backup_file, encoding="utf-8") as f:
        original = json.loads(json.load(f)["serialized_dashboard"])
    with open("dashboard_update_preview.json", encoding="utf-8") as f:
        proposed = json.load(f)

    print(f"Comparing {backup_file} (before) vs dashboard_update_preview.json (proposed)\n")

    print("--- ORIGINAL ---")
    for p in original["pages"]:
        print(f'  Page "{p["displayName"]}": {len(p.get("layout", []))} widgets')
    print(f"  Total datasets: {len(original.get('datasets', []))}")

    print("\n--- PROPOSED ---")
    for p in proposed["pages"]:
        print(f'  Page "{p["displayName"]}": {len(p.get("layout", []))} widgets')
    print(f"  Total datasets: {len(proposed.get('datasets', []))}")

    orig_pages = {p["name"]: p for p in original["pages"]}
    prop_pages = {p["name"]: p for p in proposed["pages"]}
    missing_pages = set(orig_pages) - set(prop_pages)
    print(f"\nMissing pages: {missing_pages if missing_pages else 'none'}")

    all_ok = not missing_pages
    for name, op in orig_pages.items():
        pp = prop_pages.get(name)
        if pp is None:
            continue
        orig_widget_names = {w["widget"]["name"] for w in op.get("layout", [])}
        prop_widget_names = {w["widget"]["name"] for w in pp.get("layout", [])}
        missing = orig_widget_names - prop_widget_names
        if missing:
            all_ok = False
            print(f'  Page "{name}" is MISSING widgets that existed before: {missing}')
        else:
            print(f'  Page "{name}": all {len(orig_widget_names)} original widgets still present')

    orig_ds_names = {d["name"] for d in original.get("datasets", [])}
    prop_ds_names = {d["name"] for d in proposed.get("datasets", [])}
    missing_ds = orig_ds_names - prop_ds_names
    if missing_ds:
        all_ok = False
        print(f"\nMISSING datasets that existed before: {missing_ds}")
    else:
        print(f"\nAll {len(orig_ds_names)} original datasets still present")

    print("\n" + ("SAFE TO --confirm" if all_ok else "DO NOT --confirm -- something's missing, tell Claude what printed above"))


if __name__ == "__main__":
    main()
