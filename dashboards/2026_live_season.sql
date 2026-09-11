-- 2026 Live Season -- Databricks SQL / Lakeview dashboard queries
--
-- Source tables (all written by notebooks/09_update_2026_season.py):
--   nfl_prediction_engine.fantasy_season_2026               -- one row per team per completed week
--   nfl_prediction_engine.fantasy_standings_2026             -- one row per team, season-to-date totals
--   nfl_prediction_engine.fantasy_predictions_2026           -- predicted vs actual per matchup, completed weeks only
--   nfl_prediction_engine.fantasy_championship_tracker_2026  -- one row per completed week: current top-6 seeds + predicted champion
--   nfl_prediction_engine.fantasy_bracket_2026               -- predicted vs actual playoff bracket (won't exist until weeks 14-16 resolve)
--
-- Same table shapes as the 2025 recap tables (09 was built to mirror 08's
-- output), so widgets 1-6 below are near-identical to the 2025 dashboard,
-- just pointed at the _2026 tables. Widget 7 (Championship Tracker) is new
-- -- there's no 2025 equivalent, since that table only makes sense as a
-- week-over-week live thing.
--
-- IMPORTANT: none of these tables exist until 09_update_2026_season.py has
-- found at least one completed week (it explicitly skips writing before
-- that). Every query here will legitimately return zero rows right now --
-- that's expected, not a bug. Nothing needs to change later: as your
-- scheduled jobs run through the season, these fill in on their own.


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
FROM nfl_prediction_engine.fantasy_standings_2026
ORDER BY wins DESC, points_for DESC;


-- ============================================================
-- Widget 2: Wins by Team  (Bar chart -- X: team, Y: wins)
-- ============================================================
SELECT team, wins, losses
FROM nfl_prediction_engine.fantasy_standings_2026
ORDER BY wins DESC;


-- ============================================================
-- Widget 3: Weekly Points Trend  (Line chart -- X: week, Y: points_for,
--           Group/Series: team -- label the Y axis "Points" like the 2025 page)
-- ============================================================
SELECT team, week, points_for
FROM nfl_prediction_engine.fantasy_season_2026
WHERE result != 'BYE'
ORDER BY team, week;


-- ============================================================
-- Widget 4: Matchup Prediction Accuracy So Far  (Counter / Big Number)
-- ============================================================
SELECT
  ROUND(100.0 * SUM(CASE WHEN correct THEN 1 ELSE 0 END) / COUNT(*), 1) AS accuracy_pct,
  SUM(CASE WHEN correct THEN 1 ELSE 0 END) AS correct_predictions,
  COUNT(*) AS total_predictions
FROM nfl_prediction_engine.fantasy_predictions_2026;


-- ============================================================
-- Widget 5: Prediction Accuracy by Week  (Bar or line chart --
--           X: week, Y: accuracy_pct)
-- ============================================================
SELECT
  week,
  ROUND(100.0 * SUM(CASE WHEN correct THEN 1 ELSE 0 END) / COUNT(*), 1) AS accuracy_pct,
  COUNT(*) AS games
FROM nfl_prediction_engine.fantasy_predictions_2026
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
FROM nfl_prediction_engine.fantasy_predictions_2026
ORDER BY week, team;


-- ============================================================
-- Widget 7: Championship Tracker -- Predicted Champion Over Time
-- (NEW -- no 2025 equivalent) (Table, or a line/step chart if you want to
-- visualize seed movement -- the predicted_champion column is the headline)
-- One row per completed week: "if the playoffs started today" seeding and
-- who the model would pick to win it all with that seeding.
-- ============================================================
SELECT
  as_of_week,
  seed1, seed2, seed3, seed4, seed5, seed6,
  predicted_champion
FROM nfl_prediction_engine.fantasy_championship_tracker_2026
ORDER BY as_of_week;


-- ============================================================
-- Widget 7b: Predicted Champion Only  (Line/step chart -- X: as_of_week,
--            Y: predicted_champion as a categorical series -- shows at a
--            glance whether the pick has been stable or has flip-flopped)
-- Optional alternative/companion to 7 if you want a simpler visual.
-- ============================================================
SELECT as_of_week, predicted_champion
FROM nfl_prediction_engine.fantasy_championship_tracker_2026
ORDER BY as_of_week;


-- ============================================================
-- Widget 8: Playoff Bracket -- Predicted vs. Actual  (Table or pivot,
-- same layout as the 2025 page's bracket panel)
-- Will be empty until weeks 14-16 resolve -- that's expected, not broken.
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
FROM nfl_prediction_engine.fantasy_bracket_2026
ORDER BY
  scenario,
  CASE round WHEN 'Round 1' THEN 1 WHEN 'Semifinal' THEN 2 WHEN 'Championship' THEN 3 END;
