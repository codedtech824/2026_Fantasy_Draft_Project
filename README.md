# 2026 NFL Prediction Engine & Draft Board

A Medallion Architecture (Bronze → Silver → Gold) fantasy football pipeline: ingests
real NFL data, engineers predictive features, trains an ML model, and produces a
Value-Based-Drafting (VBD) draft board — built to run on Databricks.

## Architecture

```
Bronze (raw)  →  Silver (cleaned)  →  Gold (features)  →  ML  →  Draft Board
```

All business logic lives in `src/` as plain Python classes with no Spark
dependency — they run identically locally or on a cluster. The `notebooks/`
are thin Databricks orchestration wrappers around those same classes: **logic
in `src/`, orchestration in notebooks.**

### Bronze — raw ingestion (`src/fetcher.py`, `NFLDataFetcher`)

| Source | What | Method |
|---|---|---|
| nfldata.org | Player-season stats, 2022–2025 | `fetch_nfl_data_stats` |
| nfldata.org | 2026 regular-season schedule (272 games) | `fetch_schedule` |
| nfldata.org + player stats | Team defense (D/ST) — no dedicated endpoint, so this is assembled from per-player defensive counting stats + points allowed | `fetch_dst_stats` |
| nflverse | Current 2026 roster — real team assignments, reliable ACT/CUT/DEV/RES/RET/EXE status | `fetch_nflverse_roster` |
| LeagueLogs | Market values, secondary player list (fallback) | `fetch_league_logs_data` |
| Muffed.ai | Advanced EPA metrics | `fetch_muffed_metrics` |

### Silver — cleaned & conformed (`src/bronze_to_silver.py`, `BronzeToSilver`)

- `process_stats()` → `game_logs.parquet` — one row per player-season, identity resolved into a `conformed_id`
- `process_dst()` → `dst_logs.parquet` — team defenses scored (sacks, INTs, fumble recoveries, def TDs, safeties, tiered points-allowed), merged into `game_logs.parquet`
- `process_schedule()` → `schedule_2026.parquet` — one row per team per week it plays (no bye-week logic here — that's a Gold-layer fact, not a cleaned record)
- `create_players_master()` → `players_master.parquet` — built from nflverse, deduped by `conformed_id` (the first-initial.lastname scheme collides for some real players, e.g. Bijan Robinson vs. Brian Robinson Jr. — collisions are dropped to one row each rather than risk misattributing a projection)

### Gold — feature engineering (`src/silver_to_gold.py`, `SilverToGold`)

1. **EWMA time-decay** — recent seasons weighted more (2025 = 1.0 → 2022 = 0.3)
2. **SOS normalization** — historical adjustment
3. **Injury risk** — severity-based multiplier
4. **Matchup engine** — real 2026 schedule lookup: each player's `schedule_modifier` reflects the EWMA-weighted defensive strength of the opponents they'll actually face (clipped to ±15%), and `bye_week` is looked up directly as metadata, not folded into the point total

### ML — projection (`src/predictor.py`, `NFLPredictor`)

XGBoost regressor trained on Gold features to predict `final_2026_projection`.
Logs params/metrics/model to MLflow when it's available (skips gracefully
otherwise, e.g. running locally without mlflow installed). `bye_week` and
`schedule_modifier` ride through to the output so the reasoning is visible,
not just an invisible multiplier.

### Draft Board — VBD (`src/drafter.py`, `NFLDrafter`)

- Filters to `roster_status == 'ACT'` (D/ST exempted) — keeps retired/inactive players out automatically
- Computes **data-driven replacement-level baselines** per position (the actual projected value of the last startable player for a 13-team league, not a hardcoded guess)
- Applies a scarcity multiplier and sorts into `final_draft_board.parquet`

## Running on Databricks

1. Add this repo as a **Databricks Repo** (Repos → Add Repo).
2. Run `notebooks/00_setup.py` once per new cluster session — installs
   libraries, wires up `sys.path`, creates the `/tmp` working folders (DBFS
   root is disabled on this workspace).
3. Run `notebooks/run_pipeline.py` for the full end-to-end rebuild in one
   notebook. It also registers two Delta tables so the output is queryable
   with `%sql` instead of only through `pd.read_parquet`:
   - `nfl_prediction_engine.final_predictions`
   - `nfl_prediction_engine.draft_board_2026`

   Or run `01`–`05` individually to step through each stage.

4. For the in-season update — separate from the historical rebuild —
   `notebooks/06_update_2026_stats.py` fetches stats for whatever 2026
   games have actually completed and **upserts** them via a Delta `MERGE`
   into `nfl_prediction_engine.stats_2026`, keyed so re-running it never
   duplicates a game. Safe to run weekly as the season progresses. Its
   scoring formula is fully configurable (`src/scoring.py`), not hardcoded.

5. For dashboard-ready data before the 2026 season has enough completed
   games of its own: `notebooks/07_draft_league.py` runs a 13-team, 15-round
   snake draft against the current board (`src/league.py`, fixed seed —
   reproducible, not reshuffled every run, and following a round-based
   drafting strategy — RB/WR early, QB mid-draft, K/DST last — rather than
   pure best-value), then `notebooks/08_simulate_2025_season.py` backtests
   those same rosters week-by-week against the real, completed 2025 season.
   Offense, D/ST (full formula — sacks/INTs/fumbles/def TDs/safeties +
   tiered points-allowed), and K (real FG-distance-bucket + PAT scoring)
   are all real per-week numbers from nflverse, not approximations.
   Also predicts every matchup and the playoff bracket (top 6 seeds, byes
   for #1-#2, weeks 14-16), grading those predictions against the real,
   fully-known 2025 results — the cleanest place to see how accurate a
   simple projection-based prediction actually is, honestly (52.6% matchup
   accuracy and a missed championship pick on the current data, not tuned
   to look good). Produces five Delta tables: `draft_picks_2026`,
   `fantasy_season_2025` (one row per team per week — a weekly trend
   chart), `fantasy_standings_2025`, `fantasy_predictions_2025` (predicted
   vs actual per matchup, with accuracy), and `fantasy_bracket_2025`
   (predicted vs actual bracket, two independent scenarios since they can
   diverge after Round 1).

6. Once the real 2026 season is underway, `notebooks/09_update_2026_season.py`
   is the live counterpart to `08` — scores the same drafted rosters against
   whatever 2026 games have actually completed so far (same round-robin
   schedule), producing `fantasy_season_2026`, `fantasy_standings_2026`,
   and `fantasy_predictions_2026`. It also logs a `fantasy_championship_tracker_2026`
   row every week (Delta MERGE keyed on the week, so re-running the same
   week updates it rather than duplicating) — the current top-6 seeds and
   predicted champion, so you can watch the prediction shift as the season
   plays out. Once the regular season ends and playoff weeks start
   resolving, it also produces `fantasy_bracket_2026` the same way `08`
   does. Safe to re-run anytime, e.g. weekly alongside `06`: everything
   except the tracker always recomputes from scratch, so a stat correction
   on a past week shows up immediately. Before Week 1 finishes it finds
   nothing completed and skips writing rather than create empty tables.

7. `notebooks/10_predict_nfl_games.py` is a different prediction target --
   real NFL games (Chiefs @ Broncos, etc.), not fantasy matchups. Simple
   baseline, no new modeling: predicted score is a team's rostered
   QB/RB/WR/TE `ml_projected_points` summed, minus the opponent's D/ST
   `ml_projected_points`. Validated at 59.9% accuracy against the real,
   fully-resolved 2025 season (272 games). Only depends on the draft board
   (`run_pipeline.py`), not `07`'s fantasy rosters. Produces
   `nfl_game_predictions_2026` (every game, played or not) and
   `nfl_game_predictions_graded_2026` (completed games only, with
   accuracy) -- safe to re-run anytime, e.g. on the same schedule as `06`
   and `09`.

## Running locally

```bash
pip install -r requirements.txt
python main.py          # full pipeline, writes to data/
python test_pipeline.py # automated end-to-end smoke test (20 checks)
```

## Known limitations

- No kicker/IDP scoring is wired into the draft board's roster logic by
  default, even though the data is fetched — the roster format used in
  `drafter.py`'s baselines targets QB/RB/WR/TE/DST.
- SOS normalization (Gold step 2) is a flat historical adjustment, not yet
  opponent-aware the way the matchup engine is.
- Injury data (`fetch_injuries_via_leaguelogs`) is currently sparse; most
  players get a neutral injury multiplier.
