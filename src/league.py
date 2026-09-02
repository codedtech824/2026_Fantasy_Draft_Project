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


def standard_strategy(round_num):
    """
    Default round-based drafting strategy -- a preference, not a hard rule
    (run_draft always falls back to filling a genuine roster need even if
    it means going off-strategy, so this can never cause a slot to go
    unfilled). Reflects common real-draft wisdom:

    - Rounds 1-5 (anchor talent): RB/WR/TE only. RBs and WRs are the
      scarcest, highest-variance-in-value positions, so lock those in first;
      an elite TE is worth taking here too since the drop-off after the top
      few is steep. QB is intentionally excluded -- a startable QB is
      findable much later, so spending an early pick there is a real
      opportunity cost. (The usual exception is an elite dual-threat QB
      worth a rare early reach -- not modeled here, since this board's own
      QB-vs-other-positions value scale isn't reliable enough to detect
      "elite" from, and a blanket allowance would reopen the exact
      round-1-QB problem this strategy exists to fix.)
    - Rounds 6-13 (core, flex, bench upside): QB opens up here (the
      "wait til the middle rounds" window), alongside continued RB/WR/TE.
    - Rounds 14-15 (end game): K and DST only -- late-round streamers,
      never worth an earlier pick over a skill-position player.
    """
    if round_num <= 5:
        return {"RB", "WR", "TE"}
    if round_num <= 13:
        return {"QB", "RB", "WR", "TE"}
    return {"K", "DST"}


def run_draft(board_df, snake_order, starter_slots=STARTER_SLOTS, bench_slots=BENCH_SLOTS, strategy=standard_strategy):
    """
    Greedy autodraft: at each pick, the team takes the highest-ranked
    remaining player (board_df must already be sorted descending by
    whatever value column matters) who fills one of its still-open starter
    slots. Once every starter slot is filled, remaining picks go to bench,
    still best-remaining-player-first, regardless of position. If a
    specifically-needed position has no supply left in the pool, the team
    doesn't get stuck -- it falls through to bench for that pick instead
    (matters for scarce positions like DST/K/TE).

    `strategy(round_num) -> set of positions` layers a round-based
    preference on top of pure best-value-available (default:
    standard_strategy -- pass None to disable and draft purely by value,
    the old behavior). It's a soft preference: if nothing both fills a
    genuine roster need AND matches the strategy's preferred positions this
    round, the pick falls back to whatever fills the need regardless of
    strategy, so a real need is never left unfilled just to stay on-plan.

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
        starter_needed = open_starter_positions(team) & available_positions
        preferred = strategy(p["round"]) if strategy else None

        # Priority: (1) a strategy-preferred position that also fills a real
        # starter need, (2) any strategy-preferred position at all -- lets a
        # team take a bench-quality skill player instead of being forced
        # into an early K/DST just because that's its only open *starter*
        # slot left, (3) a real starter need regardless of strategy -- only
        # reached if the preferred positions are completely gone from the
        # pool, (4) anything at all, for bench.
        if preferred:
            needed = (starter_needed & preferred) or (preferred & available_positions) or starter_needed or available_positions
        else:
            needed = starter_needed or available_positions

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
