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


def fetch_weekly_dst_points_allowed(season, session=None):
    """
    Real week-by-week D/ST scoring, points-allowed component only. Sacks,
    interceptions, fumble recoveries, and defensive TDs aren't available at
    week granularity from this API (nfldata.org's /stats/season is a
    season-total endpoint) -- those are captured in the season-total
    fetch_dst_stats/dst_logs.parquet, but not here. Points allowed is
    typically the largest and most week-to-week-variable component, so this
    is a real, if incomplete, weekly signal rather than a flat average.
    """
    session = session or requests.Session()
    session.headers.setdefault("User-Agent", "NFL-Fantasy-Pipeline/1.0")
    resp = session.get(
        "https://api.nfldata.org/v1/games",
        params={"season": season, "game_type": "REG", "limit": 500},
    )
    resp.raise_for_status()
    games = resp.json().get("data", [])

    rows = []
    for g in games:
        if g.get("home_score") is None or g.get("away_score") is None:
            continue
        week = g.get("week")
        home = _TEAM_ALIASES.get(g.get("home_team"), g.get("home_team"))
        away = _TEAM_ALIASES.get(g.get("away_team"), g.get("away_team"))
        rows.append({"team": home, "week": week,
                     "fantasy_points": BronzeToSilver._points_allowed_tier(g["away_score"])})
        rows.append({"team": away, "week": week,
                     "fantasy_points": BronzeToSilver._points_allowed_tier(g["home_score"])})
    return pd.DataFrame(rows)


def fetch_weekly_offense(season, scoring=None):
    """QB/RB/WR/TE real per-week fantasy points, reusing the same fetch this
    project already uses for the in-season 2026 updater -- it's already
    parameterized by season."""
    updater = Stats2026Updater(spark=None, season=season, scoring=scoring)
    return updater.build_stats_dataframe()


def kicker_weekly_average(game_logs, season):
    """
    No week-level kicker data exists from this API (fg_made etc. are only
    on the season-total /stats/season payload). Approximates each kicker's
    per-week score as their season total divided by games played -- a flat
    rate repeated every week they weren't on bye, not real week-to-week
    variance. Returns {conformed_id: avg_points_per_game}.
    """
    k = game_logs[(game_logs["season"] == season) & (game_logs["position"] == "K")].copy()
    if k.empty:
        return {}
    games = k["games"].replace(0, pd.NA)
    avg = (k["fantasy_points"] / games).fillna(0)
    return dict(zip(k["conformed_id"], avg))


def simulate_season(rosters, weekly_offense, weekly_dst, kicker_avg, schedule):
    """
    rosters: {team: {slot: [player dict with conformed_id/nfl_team]}}, from
      src.league.run_draft.
    weekly_offense: DataFrame from fetch_weekly_offense (conformed_id, week,
      fantasy_points).
    weekly_dst: DataFrame from fetch_weekly_dst_points_allowed (team, week,
      fantasy_points).
    kicker_avg: dict from kicker_weekly_average.
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

    def team_week_score(team, week):
        total = 0.0
        r = rosters[team]
        for slot in STARTER_SLOT_ORDER:
            for player in r.get(slot, []):
                if slot == "DST":
                    total += dst_lookup.get((player.get("team"), week), 0.0)
                elif slot == "K":
                    total += kicker_avg.get(player.get("conformed_id"), 0.0)
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
