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
    weekly_offense = weekly_offense.copy()
    if "conformed_id" not in weekly_offense.columns:
        weekly_offense["conformed_id"] = (
            weekly_offense["player_name"].str.strip().str.lower() + "_" +
            weekly_offense["position"].str.strip().str.lower()
        )
    offense_lookup = weekly_offense.set_index(["conformed_id", "week"])["fantasy_points"].to_dict()
    dst_lookup = weekly_dst.set_index(["team", "week"])["fantasy_points"].to_dict()
    kicker_lookup = weekly_kicker.set_index(["conformed_id", "week"])["fantasy_points"].to_dict()

    def team_week_score(team, week):
        total = 0.0
        r = rosters[team]
        for slot in STARTER_SLOT_ORDER:
            for player in r.get(slot, []):
                if slot == "DST":
                    total += dst_lookup.get((player.get("team"), week), 0.0)
                elif slot == "K":
                    total += kicker_lookup.get((player.get("conformed_id"), week), 0.0)
                else:
                    total += offense_lookup.get((player.get("conformed_id"), week), 0.0)
        return round(total, 2)

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
