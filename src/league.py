import random

FANTASY_TEAMS = [
    "Master Cheif", "The Halo Guys", "Chiefs and Recreation", "Halo There",
    "Halo, Is This Thing On?", "The Flood Zone", "Flood Warning",
    "Cortana's Cookies", "The Warthogs", "Warthoggin'", "Grunt Work",
    "Grunt Problems", "Elite Problems",
]

# 9 starters (QB, RB x2, WR x2, TE, FLEX, DST, K) + 6 bench = 15 rounds
STARTER_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1}
BENCH_SLOTS = 6
FLEX_ELIGIBLE = {"RB", "WR", "TE"}
DRAFT_POSITIONS = ["QB", "RB", "WR", "TE", "DST", "K"]
TOTAL_ROSTER_SIZE = sum(STARTER_SLOTS.values()) + BENCH_SLOTS


def generate_snake_order(teams=FANTASY_TEAMS, rounds=TOTAL_ROSTER_SIZE, seed=None):
    """Randomize round 1 order, then snake it (reverse each subsequent round)
    for the given number of rounds. Returns a list of
    {round, pick, team} dicts, pick numbers 1-indexed overall."""
    order = list(teams)
    random.Random(seed).shuffle(order)
    picks = []
    pick_num = 1
    for rnd in range(1, rounds + 1):
        round_order = order if rnd % 2 == 1 else list(reversed(order))
        for team in round_order:
            picks.append({"round": rnd, "pick": pick_num, "team": team})
            pick_num += 1
    return picks


def _empty_roster():
    return {pos: [] for pos in list(STARTER_SLOTS) + ["BN"]}


def run_draft(board_df, snake_order, starter_slots=STARTER_SLOTS, bench_slots=BENCH_SLOTS):
    """
    Greedy autodraft: at each pick, the team takes the highest-ranked
    remaining player (board_df must already be sorted descending by
    whatever value column matters) who fills one of its still-open starter
    slots. Once every starter slot is filled, remaining picks go to bench,
    still best-remaining-player-first, regardless of position. If a
    specifically-needed position has no supply left in the pool, the team
    doesn't get stuck -- it falls through to bench for that pick instead
    (matters for scarce positions like DST/K/TE).

    board_df: DataFrame with at least conformed_id, player_name, position,
    team, bye_week columns, one row per draftable player/DST unit.
    snake_order: output of generate_snake_order().

    Returns (rosters, pick_log) -- rosters: {team: {slot: [player dict]}},
    pick_log: list of {round, pick, team, player, position, slot} dicts in
    draft order.
    """
    available = board_df.to_dict("records")
    teams = sorted({p["team"] for p in snake_order})
    rosters = {t: _empty_roster() for t in teams}

    def open_starter_positions(team):
        r = rosters[team]
        needed = {pos for pos, n in starter_slots.items() if pos != "FLEX" and len(r[pos]) < n}
        if len(r["FLEX"]) < starter_slots.get("FLEX", 0):
            needed |= FLEX_ELIGIBLE
        return needed

    def assign(team, player):
        r = rosters[team]
        pos = player["position"]
        if pos in starter_slots and pos != "FLEX" and len(r[pos]) < starter_slots[pos]:
            r[pos].append(player)
            return pos
        if pos in FLEX_ELIGIBLE and len(r["FLEX"]) < starter_slots.get("FLEX", 0):
            r["FLEX"].append(player)
            return "FLEX"
        if len(r["BN"]) < bench_slots:
            r["BN"].append(player)
            return "BN"
        return None

    pick_log = []
    total_slots = sum(starter_slots.values()) + bench_slots
    for p in snake_order:
        team = p["team"]
        if sum(len(v) for v in rosters[team].values()) >= total_slots or not available:
            continue

        available_positions = {pl["position"] for pl in available}
        needed = open_starter_positions(team) & available_positions
        if not needed:
            needed = available_positions  # starters unfillable from remaining pool -- take best for bench

        idx = next((i for i, pl in enumerate(available) if pl["position"] in needed), None)
        if idx is None:
            continue

        player = available.pop(idx)
        slot = assign(team, player)
        pick_log.append({"round": p["round"], "pick": p["pick"], "team": team,
                          "player": player["player_name"], "position": player["position"],
                          "nfl_team": player.get("team"), "bye_week": player.get("bye_week"),
                          "conformed_id": player.get("conformed_id"), "slot": slot})

    return rosters, pick_log
