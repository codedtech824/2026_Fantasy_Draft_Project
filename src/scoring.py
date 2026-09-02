from dataclasses import dataclass


@dataclass
class ScoringSystem:
    """Point values per stat category. Defaults match FantasyData's standard
    PPR system (fantasydata.com/api/fantasy-scoring-system/nfl):
    https://fantasydata.com/api/fantasy-scoring-system/nfl

    two_point_conversion, fumble_lost, and fumble_return_td are included for
    completeness but currently contribute 0 -- nfldata.org's stat endpoints
    don't expose fumble or 2pt-conversion fields, so there's nothing to
    multiply them against yet.
    """

    # Passing
    pass_yards_per_point: float = 25.0
    pass_td: float = 4.0
    interception: float = -2.0

    # Rushing
    rush_yards_per_point: float = 10.0
    rush_td: float = 6.0

    # Receiving
    reception: float = 1.0  # 1.0 = full PPR, 0.5 = half-PPR, 0.0 = standard
    rec_yards_per_point: float = 10.0
    rec_td: float = 6.0

    # Shared / not currently populated by the data source (see docstring)
    two_point_conversion: float = 2.0
    fumble_lost: float = -2.0
    fumble_return_td: float = 6.0

    def score(
        self,
        *,
        pass_yards=0, pass_tds=0, interceptions=0,
        rush_yards=0, rush_tds=0,
        rec_yards=0, rec_tds=0, receptions=0,
        two_pt_conversions=0, fumbles_lost=0, fumble_return_tds=0,
    ):
        points = 0.0
        points += (pass_yards or 0) / self.pass_yards_per_point
        points += (pass_tds or 0) * self.pass_td
        points += (interceptions or 0) * self.interception

        points += (rush_yards or 0) / self.rush_yards_per_point
        points += (rush_tds or 0) * self.rush_td

        points += (rec_yards or 0) / self.rec_yards_per_point
        points += (rec_tds or 0) * self.rec_td
        points += (receptions or 0) * self.reception

        points += (two_pt_conversions or 0) * self.two_point_conversion
        points += (fumbles_lost or 0) * self.fumble_lost
        points += (fumble_return_tds or 0) * self.fumble_return_td
        return round(points, 2)


FULL_PPR = ScoringSystem()
HALF_PPR = ScoringSystem(reception=0.5)
STANDARD = ScoringSystem(reception=0.0)
