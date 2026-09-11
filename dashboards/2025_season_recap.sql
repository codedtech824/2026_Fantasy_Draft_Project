-- 2025 Season Recap -- Databricks SQL / Lakeview dashboard queries
--
-- Source tables (all written by notebooks/08_simulate_2025_season.py):
--   nfl_prediction_engine.fantasy_season_2025       -- one row per team per week
--   nfl_prediction_engine.fantasy_standings_2025    -- one row per team, season totals
--   nfl_prediction_engine.fantasy_predictions_2025  -- predicted vs actual per matchup
--   nfl_prediction_engine.fantasy_bracket_2025      -- predicted vs actual playoff bracket
--
-- How to build the dashboard from this file:
--   1. In Databricks, go to SQL Editor (or the "SQL" persona in the sidebar).
--   2. For each query below: paste it in, run it, then "Save" it as a named
--      query (the name in the comment above each query is a good title).
--   3. Click "+ Visualization" under the query result and pick the chart
--      type noted in that query's comment (Table / Bar / Line / Counter).
--   4. Go to Dashboards (or Lakeview) -> Create Dashboard -> "2025 Season
--      Recap" -> add each saved visualization onto the canvas. Suggested
--      layout order matches the order below, top to bottom.
--   5. All of this data is fully resolved (the 2025 season already
--      happened), so nothing here needs a refresh schedule -- unlike the
--      2026 live-season dashboard, this one is static once built.


-- ============================================================
-- Widget 1: Final Standings  (Table)
-- ============================================================
SELECT
  team,
  wins,
  losses,
  ties,
  ROUND(points_for, 1)     AS points_for,
  ROUND(points_against, 1) AS points_against,
  ROUND(points_for - points_against, 1) AS point_diff
FROM nfl_prediction_engine.fantasy_standings_2025
ORDER BY wins DESC, points_for DESC;


-- ============================================================
-- Widget 2: Wins by Team  (Bar chart -- X: team, Y: wins)
-- ============================================================
SELECT team, wins, losses
FROM nfl_prediction_engine.fantasy_standings_2025
ORDER BY wins DESC;


-- ============================================================
-- Widget 3: Weekly Points Trend  (Line chart -- X: week, Y: points_for,
--           Group/Series: team)
-- Excludes BYE weeks (points_for is still populated on a bye, but there's
-- no opponent/result, so it's not a real "performance that week" point).
-- ============================================================
SELECT team, week, points_for
FROM nfl_prediction_engine.fantasy_season_2025
WHERE result != 'BYE'
ORDER BY team, week;


-- ============================================================
-- Widget 4: Matchup Prediction Accuracy  (Counter / Big Number)
-- ============================================================
SELECT
  ROUND(100.0 * SUM(CASE WHEN correct THEN 1 ELSE 0 END) / COUNT(*), 1) AS accuracy_pct,
  SUM(CASE WHEN correct THEN 1 ELSE 0 END) AS correct_predictions,
  COUNT(*) AS total_predictions
FROM nfl_prediction_engine.fantasy_predictions_2025;


-- ============================================================
-- Widget 5: Prediction Accuracy by Week  (Bar or line chart --
--           X: week, Y: accuracy_pct)
-- ============================================================
SELECT
  week,
  ROUND(100.0 * SUM(CASE WHEN correct THEN 1 ELSE 0 END) / COUNT(*), 1) AS accuracy_pct,
  COUNT(*) AS games
FROM nfl_prediction_engine.fantasy_predictions_2025
GROUP BY week
ORDER BY week;


-- ============================================================
-- Widget 6: Predicted vs. Actual -- Matchup Detail  (Table)
-- ============================================================
SELECT
  team,
  week,
  opponent,
  ROUND(predicted_points_for, 1) AS predicted_points_for,
  ROUND(actual_points_for, 1)    AS actual_points_for,
  predicted_result,
  actual_result,
  correct
FROM nfl_prediction_engine.fantasy_predictions_2025
ORDER BY week, team;


-- ============================================================
-- Widget 7: Playoff Bracket -- Predicted vs. Actual  (Table)
-- Two independent scenarios (see notebook 08's docstring for why they're
-- not merged game-by-game): once the brackets disagree on a round's
-- winner, the next round's matchup itself differs between them.
-- ============================================================
SELECT
  scenario,
  round,
  week,
  team_a,
  ROUND(score_a, 1) AS score_a,
  team_b,
  ROUND(score_b, 1) AS score_b,
  winner
FROM nfl_prediction_engine.fantasy_bracket_2025
ORDER BY
  scenario,
  CASE round WHEN 'Round 1' THEN 1 WHEN 'Semifinal' THEN 2 WHEN 'Championship' THEN 3 END;


-- ============================================================
-- Widget 8: Predicted vs. Actual Champion  (Counter x2, or a small table)
-- Honest by design: this will show whether the model's preseason-style
-- projection actually called the champion correctly. For the 2025
-- backtest it didn't (predicted "The Halo Guys", actual champion was
-- "Halo There") -- that's real, worth showing as-is, not something to
-- hide or adjust.
-- ============================================================
SELECT scenario, winner AS champion
FROM nfl_prediction_engine.fantasy_bracket_2025
WHERE round = 'Championship';
