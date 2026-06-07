# 1B Park Factors (Verified)
park_factor_1B = {
    "COL": 117.0, "AZ": 102.0, "MIN": 104.0, "BAL": 104.0, "CIN": 96.0,
    "NYY": 92.0, "BOS": 105.0, "PHI": 103.0, "LAD": 93.0, "HOU": 98.0,
    "TOR": 100.0, "WSH": 106.0, "LAA": 98.0, "DET": 101.0, "MIA": 102.0,
    "ATL": 105.0, "KC": 100.0, "PIT": 102.0, "NYM": 96.0, "CWS": 99.0,
    "STL": 107.0, "CLE": 97.0, "SF": 103.0, "SD": 96.0, "MIL": 95.0,
    "TB": 99.0, "CHC": 97.0, "TEX": 94.0, "SEA": 90.0,
    "ATH":100
}

# 2B Park Factors (Verified)
park_factor_2B = {
    "COL": 121.0, "AZ": 117.0, "MIN": 111.0, "BAL": 101.0, "CIN": 103.0,
    "NYY": 92.0, "BOS": 117.0, "PHI": 94.0, "LAD": 94.0, "HOU": 96.0,
    "TOR": 104.0, "WSH": 100.0, "LAA": 91.0, "DET": 92.0, "MIA": 107.0,
    "ATL": 96.0, "KC": 119.0, "PIT": 118.0, "NYM": 92.0, "CWS": 94.0,
    "STL": 106.0, "CLE": 102.0, "SF": 108.0, "SD": 88.0, "MIL": 86.0,
    "TB": 88.0, "CHC": 81.0, "TEX": 90.0, "SEA": 91.0,
    "ATH":100
}

# 3B Park Factors (Verified)
park_factor_3B = {
    "COL": 193.0, "AZ": 208.0, "MIN": 85.0, "BAL": 127.0, "CIN": 70.0,
    "NYY": 76.0, "BOS": 83.0, "PHI": 100.0, "LAD": 67.0, "HOU": 71.0,
    "TOR": 72.0, "WSH": 96.0, "LAA": 94.0, "DET": 151.0, "MIA": 133.0,
    "ATL": 92.0, "KC": 179.0, "PIT": 78.0, "NYM": 79.0, "CWS": 73.0,
    "STL": 75.0, "CLE": 51.0, "SF": 140.0, "SD": 71.0, "MIL": 90.0,
    "TB": 132.0, "CHC": 119.0, "TEX": 78.0, "SEA": 40.0,
    "ATH":100
}

# HR Park Factors (Verified)
park_factor_HR = {
    "COL": 106.0, "AZ": 93.0, "MIN": 98.0, "BAL": 111.0, "CIN": 123.0,
    "NYY": 119.0, "BOS": 84.0, "PHI": 115.0, "LAD": 128.0, "HOU": 115.0,
    "TOR": 109.0, "WSH": 98.0, "LAA": 107.0, "DET": 103.0, "MIA": 87.0,
    "ATL": 95.0, "KC": 83.0, "PIT": 80.0, "NYM": 102.0, "CWS": 97.0,
    "STL": 80.0, "CLE": 94.0, "SF": 79.0, "SD": 108.0, "MIL": 104.0,
    "TB": 96.0, "CHC": 99.0, "TEX": 88.0, "SEA": 96.0,
    "ATH":100
}

home_scalers = {
    "S%":1.002,
    "D%":1.013,
    "HR%":1.025,
    "SO%":0.976,
    "WALK%":1.029
}

def log_five_scale(batter_dict,pitcher_dict): 
    """ 
    takes as input a dict of batter stats and pitcher stats (both percents)
    in the order 
    0 -> balls in play
    1 -> home runs
    2 -> walks 
    3 -> strikeouts

    returns a dict of possible actions and probs before ball is in play, adjusted for skill between both players 
    """

    leaguewide_percents = [0.657,0.028,0.094,0.221]
    batter_list_full = list(batter_dict.values())
    pitcher_list_full = list(pitcher_dict.values())

    batter_list = [round(x / 100,2) for x in batter_list_full]
    pitcher_list = [round(x / 100,2) for x in pitcher_list_full]

    # find probs of specific outcomes
    bip_prob = round(100*batter_list[0]*pitcher_list[0]/leaguewide_percents[0])
    hr_prob = round(100*batter_list[1]*pitcher_list[1]/leaguewide_percents[1])
    walk_prob = round(100*batter_list[2]*pitcher_list[2]/leaguewide_percents[2])
    so_prob = round(100*batter_list[3]*pitcher_list[3]/leaguewide_percents[3])

    # find denominator for normalized probs 
    total_prob = bip_prob + hr_prob + walk_prob + so_prob

    return {"BIP%":round(100*bip_prob/total_prob,2),
            "HR%":round(100*hr_prob/total_prob,2),
            "WALK%":round(100*walk_prob/total_prob,2),
            "SO%":round(100*so_prob/total_prob,2)}

def counts_to_percents_dict(counts_dict: dict[any, int | float]) -> dict[any, float]:
    """
    takes as input a dictionary where the values are counts
    returns a dictionary of percents, maintaining the keys and order
    """
    total = sum(counts_dict.values())
    
    # Handle the edge case of an empty dictionary or a total sum of 0
    if total == 0:
        return {key: 0.0 for key in counts_dict}
        
    return {key: round((count / total) * 100, 2) for key, count in counts_dict.items()}

def filter_batter_stats(batter_dict):
    """
    takes as input a dict of batter stats and returns two dicts in preset order

    ** plate appearances is total, not included ** 

    dicts #1 is breakdown of ball before in play
    0 -> balls in play
    1 -> home runs
    2 -> walks 
    3 -> strikeouts

    dicts #2 is breakdown of ball once it's in play
    0 -> outs on balls in play
    1 -> singles 
    2 -> doubles
    3 -> triples 
    
    """
    # collect raw data
    plate_appearances = batter_dict["PA"]
    singles = batter_dict["1B"]
    doubles = batter_dict["2B"]
    triples = batter_dict["3B"]
    home_runs = batter_dict["HR"]
    walks = batter_dict["IBB"] + batter_dict["BB"] + batter_dict["HBP"]
    strike_outs = batter_dict["SO"]

    # calculate how many balls in play
        # balls in play + home runs + walks + strike outs = total plate appearances
    balls_in_play = plate_appearances - (home_runs + walks + strike_outs) 

    # calculate how many outs on balls in play
        # singles + doubles + triples + outs in play = total balls in play
    outs_bip = balls_in_play - (singles + doubles + triples)

    before_bip = {
        "BIP%":balls_in_play,
        "HR%":home_runs,
        "WALK%":walks,
        "SO%":strike_outs
        }
    
    after_bip = {
        "OIP%":outs_bip,
        "S%":singles,
        "D%":doubles,
        "T%":triples
    }
    
    return before_bip, after_bip

def filter_pitcher_stats(pitcher_dict):
    """
    takes as input a dict of batter stats and returns a single dict in a preset order

    ** batters faced is total, not included ** 
    
    0 -> balls in play
    1 -> home runs
    2 -> walks 
    3 -> strikeouts
    
    """
    # collect raw data 
    total_batters_faced = pitcher_dict["TBF"]
    home_runs = pitcher_dict["HR"]
    walks = pitcher_dict["BB"] + pitcher_dict["HBP"]
    strike_outs = pitcher_dict["SO"]

    # calculate how many balls in play
        # balls in play + home runs + walks + strike outs = total batters faced
    balls_in_play = total_batters_faced - (home_runs + walks + strike_outs)

    return {"BIP%":balls_in_play,
            "HR%":home_runs,
            "WALK%":walks,
            "SO%":strike_outs}

def normalize_pf(stat_dict, team_abb):
    """ 
    takes as input 
    1) a dict of stats
    2) abbreviation for the player's home team

    returns dict with normalized stats, on a neutral field 
    """
    stats = stat_dict.copy()

    # retrieve park factors
    pf_singles = round(2-park_factor_1B[team_abb]/100,2)
    pf_doubles = round(2-park_factor_2B[team_abb]/100,2)
    pf_triples = round(2-park_factor_3B[team_abb]/100,2)
    pf_home_runs = round(2-park_factor_HR[team_abb]/100,2)

    # normalize relevant stats 
    try: 
        stats["1B"] = stat_dict["1B"] + 0.5 * (stat_dict["1B"] * pf_singles - stat_dict["1B"])
    except KeyError:
        return None
    stats["2B"] = stat_dict["2B"] + 0.5 * (stat_dict["2B"] * pf_doubles - stat_dict["2B"])
    stats["3B"] = stat_dict["3B"] + 0.5 * (stat_dict["3B"] * pf_triples - stat_dict["3B"])
    stats["HR"] = stat_dict["HR"] + 0.5 * (stat_dict["HR"] * pf_home_runs - stat_dict["HR"])

    return stats

def adjust_handedness():
    ...
    # will do this once I buy the FanGraphs subscription for rightie leftie splits


def reapply_pf(stat_dicts: list[dict[str, float]], stadium_abb: str) -> list[dict[str, float]]:
    """
    Applies stadium-specific park factor adjustments to player projection percentages.
    Values are rounded to the nearest tenth, with a minimum floor of 0.005.
    """
    pf_mappings = {
        "1B%": park_factor_1B,
        "2B%": park_factor_2B,
        "3B%": park_factor_3B,
        "HR%": park_factor_HR
    }
    
    adjusted_lineup = []
    
    for player_stats in stat_dicts:
        adjusted_stats = player_stats.copy()
        
        for stat_key, pf_dict in pf_mappings.items():
            if stat_key in adjusted_stats:
                pf_value = pf_dict.get(stadium_abb, 100.0)
                scaling_factor = pf_value / 100.0
                
                # Calculate scaled value
                scaled_val = adjusted_stats[stat_key] * scaling_factor
                
                # Round to the nearest tenth place
                rounded_val = round(scaled_val, 3)
                
                # Enforce the 0.005 floor if the value rounds down to 0
                adjusted_stats[stat_key] = max(rounded_val, 0.005)
                
        adjusted_lineup.append(adjusted_stats)
        
    return adjusted_lineup

def apply_home_adv(stat_dicts: list[dict[str, float]]) -> list[dict[str, float]]:
    """
    Applies home advantage multipliers to a lineup's projected stat percentages.
    Values are rounded to the nearest tenth, with a minimum floor of 0.005.
    """
    adjusted_lineup = []
    
    for player_stats in stat_dicts:
        adjusted_stats = player_stats.copy()
        
        for stat_key in adjusted_stats:
            if stat_key in home_scalers:
                multiplier = home_scalers[stat_key]
                
                # Calculate scaled value
                scaled_val = adjusted_stats[stat_key] * multiplier
                
                # Round to the nearest tenth place
                rounded_val = round(scaled_val, 3)
                
                # Enforce the 0.005 floor if the value rounds down to 0
                adjusted_stats[stat_key] = max(rounded_val, 0.005)
                
        adjusted_lineup.append(adjusted_stats)
        
    return adjusted_lineup