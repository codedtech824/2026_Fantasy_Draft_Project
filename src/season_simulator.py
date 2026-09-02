import io

import requests
import pandas as pd

from src.bronze_to_silver import BronzeToSilver
from src.stats_updater import Stats2026Updater

STARTER_SLOT_ORDER = ["QB", "RB", "WR", "TE", "FLEX", "DST", "K"]

_TEAM_ALIASES = {"WSH": "WAS"}  # see fetch_dst_stats / process_schedule for why


def generate_round_robin(teams, weeks=None):
    """
    Standard circle-method round robin: every team plays every other team
    exactly once. Odd team counts (e.g. 13) get a bye each round via a
    placeholder seat. Returns a list of {week, team, opponent} rows (two
    rows per game, one per side) plus one {week, team, opponent: None} row
    for whichever team has the bye that week.
    """
    seats = list(teams)
    if len(seats) % 2 == 1:
        seats.append(None)
    n = len(seats)
    total_weeks = weeks or (n - 1)

    rotation = seats[:]
    schedule = []
    for week in range(1, total_weeks + 1):
        pairs = list(zip(rotation[: n // 2], reversed(rotation[n // 2 :])))
        for a, b in pairs:
            if a is None:
                schedule.append({"week": week, "team": b, "opponent": None})
            elif b is None:
                schedule.append({"week": week, "team": a, "opponent": None})
            else:
                schedule.append({"week": week, "team": a, "opponent": b})
                schedule.append({"week": week, "team": b, "opponent": a})
        rotation = [rotation[0], rotation[-1]] + rotation[1:-1]
    return schedule


def completed_weeks_for_season(season, max_week=None, session=None):
    """
    The set of real NFL week numbers that have at least one completed game
    (both scores populated) for `season`, capped at `max_week` (the fantasy
    schedule's length -- no point knowing about NFL weeks past that). Used
    to figure out how much of the fantasy schedule has actually happened
    yet, e.g. for the live 2026 season -- empty early in/before the season,
    growing by one week roughly every Tuesday.
    """
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", "NFL-Fantasy-Pipeline/1.0")
    resp = session.get(
        "https://api.nfldata.org/v1/games",
        params={"season": season, "game_type": "REG", "limit": 500},
    )
    resp.raise_for_status()
    games = resp.json().get("data", [])
    weeks = {
        g["week"] for g in games
        if g.get("home_score") is not None and g.get("away_score") is not None
    }
    if max_week is not None:
        weeks = {w for w in weeks if w <= max_week}
    return weeks


def fetch_weekly_offense(season, scoring=None):
    """QB/RB/WR/TE real per-week fantasy points, reusing the same fetch this
    project already uses for the in-season 2026 updater -- it's already
    parameterized by season."""
    updater = Stats2026Updater(spark=None, season=season, scoring=scoring)
    return updater.build_stats_dataframe()


def _fetch_nflverse_weekly_raw(season, session=None):
    """Shared raw fetch of nflverse's weekly player-stats file (all
    positions, not just offense) -- used to build real weekly K and D/ST
    scores. Offense itself goes through Stats2026Updater/src.scoring so it
    stays configurable; this covers the positions that class doesn't."""
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", "NFL-Fantasy-Pipeline/1.0")
    url = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
    resp = session.get(url)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    df["team"] = df["team"].replace(_TEAM_ALIASES)
    return df


def fetch_weekly_kicker_points(season, session=None):
    """
    Real per-week kicker scoring (FG distance buckets + PAT) -- nflverse's
    weekly file has actual week-level kicking data, unlike nfldata.org
    (which only has kicking stats at a season-total grain). Returns a
    DataFrame with conformed_id, week, fantasy_points.
    """
    df = _fetch_nflverse_weekly_raw(season, session)
    k = df[df["position"] == "K"].copy()
    if k.empty:
        return pd.DataFrame(columns=["conformed_id", "week", "fantasy_points"])

    short_fg = k[["fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49"]].fillna(0).sum(axis=1)
    long_fg = k[["fg_made_50_59", "fg_made_60_"]].fillna(0).sum(axis=1)
    k["fantasy_points"] = short_fg * 3 + long_fg * 5 + k["pat_made"].fillna(0) * 1
    k["conformed_id"] = k["player_name"].str.strip().str.lower() + "_k"
    return k[["conformed_id", "week", "fantasy_points"]]


def fetch_weekly_dst_scores(season, session=None):
    """
    Real per-week D/ST scoring: sacks/INTs/fumble recoveries/def TDs/
    safeties from nflverse's weekly file, aggregated by team, plus tiered
    points-allowed from nfldata.org's /games (nflverse's player-stats file
    doesn't carry final scores). Full formula now, not the points-allowed-
    only approximation this used before nflverse's weekly file was
    available. Returns a DataFrame with team, week, fantasy_points.
    """
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", "NFL-Fantasy-Pipeline/1.0")

    df = _fetch_nflverse_weekly_raw(season, session)
    counting = df.groupby(["team", "week"]).agg(
        def_sacks=("def_sacks", "sum"),
        def_interceptions=("def_interceptions", "sum"),
        def_fumbles=("def_fumbles", "sum"),
        def_tds=("def_tds", "sum"),
        def_safeties=("def_safeties", "sum"),
    ).reset_index()

    resp = session.get(
        "https://api.nfldata.org/v1/games",
        params={"season": season, "game_type": "REG", "limit": 500},
    )
    resp.raise_for_status()
    games = resp.json().get("data", [])
    pa_rows = []
    for g in games:
        if g.get("home_score") is None or g.get("away_score") is None:
            continue
        week = g.get("week")
        home = _TEAM_ALIASES.get(g.get("home_team"), g.get("home_team"))
        away = _TEAM_ALIASES.get(g.get("away_team"), g.get("away_team"))
        pa_rows.append({"team": home, "week": week, "points_allowed": g["away_score"]})
        pa_rows.append({"team": away, "week": week, "points_allowed": g["home_score"]})
    pa_df = pd.DataFrame(pa_rows)

    merged = counting.merge(pa_df, on=["team", "week"], how="left")
    merged["fantasy_points"] = (
        merged["def_sacks"].fillna(0) * 1
        + merged["def_interceptions"].fillna(0) * 2
        + merged["def_fumbles"].fillna(0) * 2
        + merged["def_safeties"].fillna(0) * 2
        + merged["def_tds"].fillna(0) * 6
        + merged["points_allowed"].apply(BronzeToSilver._points_allowed_tier)
    )
    return merged[["team", "week", "fantasy_points"]]


def build_weekly_lookups(weekly_offense, weekly_dst, weekly_kicker):
    """
    Turns the three weekly-stats DataFrames into {key: fantasy_points}
    lookup dicts -- shared by simulate_season() and anything else (like the
    actual-results playoff bracket) that needs a team's real score for an
    arbitrary week, not just the weeks in a fixed schedule.
    """
    weekly_offense = weekly_offense.copy()
    if "conformed_id" not in weekly_offense.columns:
        weekly_offense["conformed_id"] = (
            weekly_offense["player_name"].str.strip().str.lower() + "_" +
            weekly_offense["position"].str.strip().str.lower()
        )
    return {
        "offense": weekly_offense.set_index(["conformed_id", "week"])["fantasy_points"].to_dict(),
        "dst": weekly_dst.set_index(["team", "week"])["fantasy_points"].to_dict(),
        "kicker": weekly_kicker.set_index(["conformed_id", "week"])["fantasy_points"].to_dict(),
    }


def compute_team_week_score(rosters, team, week, lookups):
    """A team's real score for one specific week, from the lookups
    build_weekly_lookups() produces. Missing data (bye, no stats found)
    contributes 0 for that player, same as simulate_season()."""
    total = 0.0
    for slot in STARTER_SLOT_ORDER:
        for player in rosters[team].get(slot, []):
            if slot == "DST":
                total += lookups["dst"].get((player.get("team"), week), 0.0)
            elif slot == "K":
                total += lookups["kicker"].get((player.get("conformed_id"), week), 0.0)
            else:
                total += lookups["offense"].get((player.get("conformed_id"), week), 0.0)
    return round(total, 2)


def actual_scores_for_weeks(rosters, weeks, weekly_offense, weekly_dst, weekly_kicker):
    """{(team, week): points} for every team across the given weeks --
    ready to pass straight into simulate_playoff_bracket's actual_scores.
    Use this to get REAL playoff-week results for a season where that data
    already exists (e.g. 2025's actual weeks 14-16), separate from the
    fixed regular-season `schedule` a round robin produces."""
    lookups = build_weekly_lookups(weekly_offense, weekly_dst, weekly_kicker)
    return {
        (team, week): compute_team_week_score(rosters, team, week, lookups)
        for team in rosters
        for week in weeks
    }


def simulate_season(rosters, weekly_offense, weekly_dst, weekly_kicker, schedule):
    """
    rosters: {team: {slot: [player dict with conformed_id/nfl_team]}}, from
      src.league.run_draft.
    weekly_offense: DataFrame from fetch_weekly_offense (conformed_id, week,
      fantasy_points).
    weekly_dst: DataFrame from fetch_weekly_dst_scores (team, week,
      fantasy_points).
    weekly_kicker: DataFrame from fetch_weekly_kicker_points (conformed_id,
      week, fantasy_points).
    schedule: list of {week, team, opponent} from generate_round_robin.

    Returns a DataFrame: team, week, opponent, points_for, points_against,
    result (W/L/T/BYE) -- one row per team per week.
    """
    lookups = build_weekly_lookups(weekly_offense, weekly_dst, weekly_kicker)

    def team_week_score(team, week):
        return compute_team_week_score(rosters, team, week, lookups)

    weeks = sorted({row["week"] for row in schedule})
    all_teams = sorted(rosters.keys())
    scores = {(team, week): team_week_score(team, week) for team in all_teams for week in weeks}

    rows = []
    for row in schedule:
        team, week, opp = row["team"], row["week"], row["opponent"]
        pf = scores[(team, week)]
        if opp is None:
            rows.append({"team": team, "week": week, "opponent": None,
                         "points_for": pf, "points_against": None, "result": "BYE"})
            continue
        pa = scores[(opp, week)]
        result = "W" if pf > pa else ("L" if pf < pa else "T")
        rows.append({"team": team, "week": week, "opponent": opp,
                     "points_for": pf, "points_against": pa, "result": result})

    return pd.DataFrame(rows)


# ---- predictions & playoff bracket -----------------------------------------
# Everything below predicts outcomes *before* they're known, so results can
# be graded against reality afterward. The prediction itself is deliberately
# simple: each team's predicted score is the season-long average of its
# starters' ml_projected_points (from the draft board), divided by a flat
# 17-game season -- the same number every week, not re-tuned per matchup or
# opponent. It's a baseline to grade actual results against, not a
# sophisticated week-by-week model -- and since ml_projected_points itself
# used to be badly skewed toward QBs (fixed in silver_to_gold.py), these
# predictions are only as good as that fix.

_SEASON_GAMES = 17


def team_predicted_weekly_score(rosters, team):
    """Static predicted weekly score: sum of the team's starters'
    ml_projected_points (season-long projection from the draft board),
    each divided by a flat 17-game season. Same figure used for every
    week -- see module docstring above for why that's a deliberate
    simplification, not an oversight."""
    total = 0.0
    for slot in STARTER_SLOT_ORDER:
        for player in rosters[team].get(slot, []):
            proj = player.get("ml_projected_points") or 0
            total += proj / _SEASON_GAMES
    return round(total, 2)


def predict_matchup_outcomes(rosters, schedule):
    """
    Predicted winner/score for every scheduled matchup (byes excluded --
    nothing to predict). One row per team per week they play, columns:
    team, week, opponent, predicted_points_for, predicted_points_against,
    predicted_result (W/L/T).
    """
    team_scores = {team: team_predicted_weekly_score(rosters, team) for team in rosters}
    rows = []
    for row in schedule:
        team, week, opp = row["team"], row["week"], row["opponent"]
        if opp is None:
            continue
        pf, pa = team_scores[team], team_scores[opp]
        rows.append({
            "team": team, "week": week, "opponent": opp,
            "predicted_points_for": pf, "predicted_points_against": pa,
            "predicted_result": "W" if pf > pa else ("L" if pf < pa else "T"),
        })
    return pd.DataFrame(rows)


def compare_predictions_to_actual(predictions_df, actual_df):
    """
    Joins predicted vs. actual outcomes on (team, week) -- inner join, so
    only weeks that have actually been played show up (actual_df only ever
    has rows for completed weeks, from simulate_season). Adds `correct`
    (predicted_result == actual_result) and `points_for_error` (predicted
    minus actual) so accuracy is directly visible in the table, not
    something you have to compute yourself downstream.
    """
    merged = predictions_df.merge(
        actual_df[["team", "week", "points_for", "points_against", "result"]],
        on=["team", "week"], how="inner",
    )
    merged = merged.rename(columns={
        "points_for": "actual_points_for",
        "points_against": "actual_points_against",
        "result": "actual_result",
    })
    merged["correct"] = merged["predicted_result"] == merged["actual_result"]
    merged["points_for_error"] = (merged["predicted_points_for"] - merged["actual_points_for"]).round(2)
    return merged


def compute_standings(season_df):
    """Standings (rank = seed) from a season/backtest DataFrame in the same
    shape simulate_season() produces. Factored out here so the playoff
    seeder and 08/09's own standings display can share one implementation."""
    played = season_df[season_df["result"] != "BYE"]
    standings = (
        played.groupby("team")
        .agg(
            wins=("result", lambda s: (s == "W").sum()),
            losses=("result", lambda s: (s == "L").sum()),
            ties=("result", lambda s: (s == "T").sum()),
            points_for=("points_for", "sum"),
            points_against=("points_against", "sum"),
        )
        .reset_index()
    )
    standings[["points_for", "points_against"]] = standings[["points_for", "points_against"]].round(2)
    standings = standings.sort_values(["wins", "points_for"], ascending=[False, False]).reset_index(drop=True)
    standings.index += 1
    return standings


def simulate_playoff_bracket(seeds, rosters, actual_scores=None):
    """
    Top-6-with-byes bracket, 3 rounds (weeks 14-16 by convention): Round 1
    is seed3 vs seed6 and seed4 vs seed5; seeds 1-2 sit out and meet those
    winners in the semifinal; the champion is decided in week 16.

    seeds: the top 6 team names in seed order, [seed1, ..., seed6] (e.g.
    compute_standings(...)['team'].head(6).tolist()).
    actual_scores: optional {(team, week): points_for} -- for any
    (team, week) present here, the REAL score is used instead of the
    static predicted one, so a bracket for a season already partway
    through its playoffs reflects real results for the rounds that have
    actually happened and only predicts the rounds that haven't yet.

    Returns (games, champion): games is a list of {round, week, team_a,
    score_a, team_b, score_b, winner, is_predicted} dicts in play order;
    champion is the winning team name.
    """
    def score(team, week):
        if actual_scores and (team, week) in actual_scores:
            return actual_scores[(team, week)], False
        return team_predicted_weekly_score(rosters, team), True

    games = []

    def play(team_a, team_b, week, round_name):
        sa, pred_a = score(team_a, week)
        sb, pred_b = score(team_b, week)
        winner = team_a if sa >= sb else team_b
        games.append({
            "round": round_name, "week": week,
            "team_a": team_a, "score_a": sa, "team_b": team_b, "score_b": sb,
            "winner": winner, "is_predicted": pred_a or pred_b,
        })
        return winner

    seed1, seed2, seed3, seed4, seed5, seed6 = seeds
    win_36 = play(seed3, seed6, 14, "Round 1")
    win_45 = play(seed4, seed5, 14, "Round 1")
    sf1 = play(seed1, win_45, 15, "Semifinal")
    sf2 = play(seed2, win_36, 15, "Semifinal")
    champion = play(sf1, sf2, 16, "Championship")

    return games, champion


# ---- real NFL game predictions ---------------------------------------------
# Same idea as the fantasy predictor, pointed at real NFL teams instead of
# drafted rosters: a simple offense-vs-opponent's-defense baseline, not a
# real point-spread model. No new modeling -- reuses ml_projected_points
# (offense) and the D/ST row's own ml_projected_points (defense) that are
# already sitting in the draft board.

def fetch_nfl_games(season, game_type="REG", session=None):
    """The season's real schedule, played or not, with real scores where
    available -- {season, week, game_id, home_team, away_team, home_score,
    away_score}, team codes normalized the same way as everywhere else."""
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", "NFL-Fantasy-Pipeline/1.0")
    resp = session.get(
        "https://api.nfldata.org/v1/games",
        params={"season": season, "game_type": game_type, "limit": 500},
    )
    resp.raise_for_status()
    games = resp.json().get("data", [])
    for g in games:
        g["home_team"] = _TEAM_ALIASES.get(g.get("home_team"), g.get("home_team"))
        g["away_team"] = _TEAM_ALIASES.get(g.get("away_team"), g.get("away_team"))
    return games


def fetch_weekly_injuries(season, session=None):
    """
    Real per-week "Out" designations from nflverse's injury reports --
    {week: {conformed_id, ...}}, same first-initial.lastname_position key
    used everywhere else so it joins directly against the draft board.
    Returns {} if the season's file doesn't exist yet (e.g. before Week 1's
    first injury report is published) rather than raising -- the same
    "nothing yet" pattern as the rest of the in-season fetchers.
    """
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", "NFL-Fantasy-Pipeline/1.0")
    url = f"https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"
    resp = session.get(url)
    if resp.status_code == 404:
        return {}
    resp.raise_for_status()

    inj = pd.read_csv(io.StringIO(resp.text))
    inj["team"] = inj["team"].map(lambda t: _TEAM_ALIASES.get(t, t))
    inj["conformed_id"] = (
        (inj["first_name"].str[0] + "." + inj["last_name"]).str.lower().str.strip()
        + "_" + inj["position"].str.lower().str.strip()
    )
    out = inj[inj["report_status"] == "Out"]
    return out.groupby("week")["conformed_id"].apply(set).to_dict()


def _team_offense_strength(board, team, excluded_ids=None):
    """Sum of ml_projected_points across a real NFL team's rostered
    QB/RB/WR/TE -- a rough total-offensive-output proxy, nothing fancier.
    excluded_ids (from fetch_weekly_injuries) drops players ruled "Out"
    for that week -- a benched starter shouldn't count toward the total."""
    mask = (board["team"] == team) & (board["position"].isin(["QB", "RB", "WR", "TE"]))
    if excluded_ids:
        mask &= ~board["conformed_id"].isin(excluded_ids)
    return board.loc[mask, "ml_projected_points"].sum()


def _team_defense_strength(board, team):
    """A team's own D/ST row's ml_projected_points -- the same season
    projection already computed for the draft board, not a new model."""
    mask = (board["team"] == team) & (board["position"] == "DST")
    vals = board.loc[mask, "ml_projected_points"]
    return vals.iloc[0] if len(vals) else 0.0


HOME_FIELD_BONUS_PCT = 0.03  # see predict_nfl_games docstring -- backtested as accuracy-neutral, not a proven gain


def predict_nfl_games(board, games, injuries_by_week=None):
    """
    One predicted row per real NFL game: predicted_home_score/
    predicted_away_score are an offense-minus-opponent-defense proxy, not a
    real predicted point total -- whichever side is higher is
    predicted_winner. board: a draft-board-shaped DataFrame with team,
    position, ml_projected_points (e.g. final_draft_board.parquet).
    games: from fetch_nfl_games(). injuries_by_week (from
    fetch_weekly_injuries(), optional): excludes that week's "Out" players
    from the offense sum -- validated against the real 2025 season at
    +1.5 points of accuracy (59.9% -> 61.4%) even with an imperfect
    cross-season player match, so worth passing when you have it.

    The home team's offense also gets a flat HOME_FIELD_BONUS_PCT boost.
    Honesty check: this was backtested against the real 2025 season across
    bonus sizes from 2% to 10% and never beat the baseline by more than 1
    game out of 272 at any size -- it's here for real-world modeling
    completeness (home-field advantage is real), not because the backtest
    proved it helps this particular model.
    """
    rows = []
    for g in games:
        home, away, wk = g["home_team"], g["away_team"], g["week"]
        excluded = (injuries_by_week or {}).get(wk, set())
        off_home = _team_offense_strength(board, home, excluded) * (1 + HOME_FIELD_BONUS_PCT)
        off_away = _team_offense_strength(board, away, excluded)
        def_home, def_away = _team_defense_strength(board, home), _team_defense_strength(board, away)
        pred_home = round(off_home - def_away, 2)
        pred_away = round(off_away - def_home, 2)
        rows.append({
            "game_id": g["game_id"], "season": g["season"], "week": g["week"],
            "home_team": home, "away_team": away,
            "predicted_home_score": pred_home, "predicted_away_score": pred_away,
            "predicted_winner": home if pred_home >= pred_away else away,
        })
    return pd.DataFrame(rows)


def grade_nfl_predictions(predictions_df, games):
    """
    Inner-joins predictions against real results for whichever games in
    `games` have actually been played (home_score/away_score populated) --
    games not yet played simply don't appear in the output. Adds
    actual_winner and `correct`.
    """
    actual_rows = []
    for g in games:
        if g.get("home_score") is None or g.get("away_score") is None:
            continue
        if g["home_score"] > g["away_score"]:
            winner = g["home_team"]
        elif g["away_score"] > g["home_score"]:
            winner = g["away_team"]
        else:
            winner = "TIE"
        actual_rows.append({
            "game_id": g["game_id"],
            "actual_home_score": g["home_score"], "actual_away_score": g["away_score"],
            "actual_winner": winner,
        })

    if not actual_rows:
        return predictions_df.iloc[0:0].assign(
            actual_home_score=pd.Series(dtype=float), actual_away_score=pd.Series(dtype=float),
            actual_winner=pd.Series(dtype=str), correct=pd.Series(dtype=bool),
        )

    merged = predictions_df.merge(pd.DataFrame(actual_rows), on="game_id", how="inner")
    merged["correct"] = merged["predicted_winner"] == merged["actual_winner"]
    return merged


def decompose_score(points):
    """
    Breaks a target point total into a plausible NFL scoring combination:
    touchdown+PAT=7, TD+2pt=8, TD only (missed PAT)=6, field goal=3.
    Prefers more touchdowns over more field goals, and standard PATs over
    2pt conversions/missed PATs (at most 2 non-standard-PAT touchdowns),
    matching how a real game usually breaks down. Not a unique or
    "correct" decomposition -- there are many real combinations for most
    totals -- just one that reads as plausible.
    """
    points = max(0, round(points))
    for tds in range(points // 6, -1, -1):
        for irregular in range(0, min(tds, 2) + 1):
            for two_pt in range(0, irregular + 1):
                missed = irregular - two_pt
                pat = tds - irregular
                remainder = points - (tds * 6 + pat + two_pt * 2)
                if remainder >= 0 and remainder % 3 == 0:
                    return {
                        "touchdowns": tds, "extra_points": pat,
                        "two_point_conversions": two_pt, "missed_extra_points": missed,
                        "field_goals": remainder // 3,
                    }
    return {"touchdowns": 0, "extra_points": 0, "two_point_conversions": 0,
            "missed_extra_points": 0, "field_goals": points // 3}


def add_realistic_scores(predictions_df, target_min=10, target_max=34):
    """
    predicted_home_score/predicted_away_score are an abstract comparison
    number (offense minus opponent defense, both season-long
    ml_projected_points sums) -- useful for picking a winner, not anything
    resembling a real final score (values run into the hundreds or more).
    This rescales them into a realistic NFL point range using the min/max
    across every predicted score in the DataFrame (so a team's *relative*
    strength is preserved -- the best predicted offense still lands the
    highest realistic score), then decomposes each side's realistic score
    into touchdowns/PATs/2pt conversions/field goals via decompose_score().

    Adds, per side (home/away): *_realistic_score and the
    *_touchdowns/*_extra_points/*_two_point_conversions/
    *_missed_extra_points/*_field_goals columns from decompose_score().
    """
    df = predictions_df.copy()
    all_scores = pd.concat([df["predicted_home_score"], df["predicted_away_score"]])
    lo, hi = all_scores.min(), all_scores.max()
    span = hi - lo if hi > lo else 1

    def rescale(raw):
        return target_min + (raw - lo) / span * (target_max - target_min)

    for side in ("home", "away"):
        realistic = df[f"predicted_{side}_score"].apply(rescale)
        df[f"{side}_realistic_score"] = realistic.round().astype(int)
        parts = df[f"{side}_realistic_score"].apply(decompose_score)
        for key in ("touchdowns", "extra_points", "two_point_conversions", "missed_extra_points", "field_goals"):
            df[f"{side}_{key}"] = parts.apply(lambda p: p[key])

    return df
