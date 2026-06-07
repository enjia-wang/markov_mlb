import pandas as pd
import mlb_stat_collection 
import mlb_stat_adjust
import random 
import matplotlib.pyplot as plt
import statistics

"""
1) no runners 
2) 1 runner, on first 
3) 1 runner, on second 
4) 1 runner, on third
5) 2 runners, on first and second
6) 2 runners, on first and third
7) 2 runners, on second and third
8) 3 runners
"""

"""
other props worth investigating:
1) strike outs by pitcher
2) hits by player 
3) hits + runs + rbis (runs batted in)
4) total bases

"""

"""
!!! BETTING RULES !!! 

overall moneyline - under 0.375, over 0.575 until 1.000, except 0.625-0.650
    systemically underestimates the abilities of the home team 
overall over/under - over 0.600, except 0.650-0.725
    under probabilities are pretty accurate
    tends to overestimate over probabilities, which fits the higher avg runs 
fifth inning over/under - under 0.375, over 0.600 until 1.000
RIFI - under 0.4 and above 0.6

"""

# list of samples considered out of play
out_of_play = ["HR%","SO%","WALK%"]

# list of states considered absorbing (3 outs)
absorbing_states = []

def check_double_play():
    return random.choices(
        population=[True,False],
        weights=[11,89],
        k=1
    )[0]

def check_score_on_out():
    return random.choices(
        population=[True,False],
        weights=[70,30],
        k=1
    )[0]

def check_advance_on_out():
    return random.choices(
        population=[True,False],
        weights=[25,75],
        k=1
    )[0]

def create_and_save_histogram(data: list[int], filename: str) -> None:
    """
    Creates a histogram from a list of discrete integers, displays the mean 
    and standard deviation in a corner box, and saves it as an image.

    Args:
        data (list[int]): A list of discrete integer values.
        filename (str): The name of the output image file (e.g., 'histogram.png').
    """
    if not data:
        print("Error: The data list is empty.")
        return

    # 1. Calculate statistics
    mean_val = statistics.mean(data)
    # Standard deviation requires at least two data points
    if len(data) > 1:
        std_val = statistics.stdev(data)
        stats_text = f"Mean: {mean_val:.2f}\nStd Dev: {std_val:.2f}"
    else:
        stats_text = f"Mean: {mean_val:.2f}\nStd Dev: N/A"

    # 2. Define bins for discrete integers
    min_val = min(data)
    max_val = max(data)
    bins = range(min_val, max_val + 2)

    # 3. Set up and draw the plot
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, align='left', edgecolor='black', color='skyblue')

    # 4. Add formatting and labels
    plt.title('Histogram of Discrete Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xticks(range(min_val, max_val + 1))

    # 5. Add the statistics box in the top-right corner
    # bbox creates the physical box around the text. 
    # transAxes uses relative coordinates (0 to 1) rather than data coordinates.
    box_properties = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)
    plt.gca().text(0.95, 0.95, stats_text, 
                   transform=plt.gca().transAxes, 
                   fontsize=11, 
                   verticalalignment='top', 
                   horizontalalignment='right', 
                   bbox=box_properties)

    # 6. Save and close
    plt.savefig(filename, bbox_inches='tight')
    print(f"Histogram successfully saved to '{filename}'")
    plt.close()

def transition_states(state, action): 
    """ 
    input 1: state, which is a tuple with 2 ints
    input 2: action, which is a single string

    output 1: new state after action
    output 2: number of runs scored during action

    first entry in tuple is 1-8, representing batters on base
    second entry in tuple is 0-3, representing number of outs

    normally, all runners advance up to the action, with 4 exceptions
    1) when a runner is on first, less than two outs, ball is out in play; chance for double play 
    2) when a runner is on third, ball is out; chance to advance
    *** third possibility of lone runner on first or second advancing two bases
    *** fourth possibility lead runner, ball is out; chance for them alone to advance

    couldn't find probabilities to implement 3) and 4) 
    """
    # count how many outs there are
    outs = state[1]
    
    # toggable, make all third bases score on out
    # score_on_out = True

    # action is walk
    if action == "WALK%":
        if state[0] == 1: 
            return (2,outs), 0
        elif state[0] == 2 or state[0] == 3: 
            return (5,outs), 0
        elif state[0] == 4: 
            return (6,outs), 0
        elif state[0] == 5 or state[0] == 6 or state[0] == 7: 
            return (8,outs), 0
        elif state[0] == 8:
            return (8,outs), 1
         
    # action is single
    if action == "S%":
        if state[0] == 1: 
            return (2,outs), 0
        elif state[0] == 2: 
            return (5,outs), 0
        elif state[0] == 3: 
            return (6,outs), 0
        elif state[0] == 4: 
            return (2,outs), 1
        elif state[0] == 5: 
            return (8,outs), 0
        elif state[0] == 6: 
            return (3,outs), 1
        elif state[0] == 7: 
            return (5,outs), 1
        elif state[0] == 8:
            return (8,outs), 1
        
    # action is double
    if action == "D%": 
        if state[0] == 1: 
            return (3,outs), 0
        elif state[0] == 2: 
            return (6,outs), 0
        elif state[0] == 3 or state[0] == 4: 
            return (3,outs), 1
        elif state[0] == 5 or state[0] == 6: 
            return (7,outs), 1
        elif state[0] == 7: 
            return (3,outs), 2
        elif state[0] == 8:
            return (7,outs), 2       

    # action is triple
    if action == "T%": 
        if state[0] == 1: 
            return (4,outs), 0
        elif state[0] == 2 or state[0] == 3 or state[0] == 4: 
            return (4,outs), 1
        elif state[0] == 5 or state[0] == 6 or state[0] == 7: 
            return (4,outs), 2
        elif state[0] == 8:
            return (4,outs), 3  
        
    # action is home run 
    if action == "HR%": 
        if state[0] == 1: 
            return (1,outs), 1
        elif state[0] == 2 or state[0] == 3 or state[0] == 4: 
            return (1,outs), 2
        elif state[0] == 5 or state[0] == 6 or state[0] == 7:
            return (1,outs), 3
        elif state[0] == 8: 
            return (1,outs), 4

    # action is strike out 
    if action == "SO%": 
        return (state[0],outs+1), 0

    # action is out in play 
    if action == "OIP%": 
        # print(state[0])
        if state[0] == 1: 
            return (1,outs+1), 0
        
        elif state[0] == 2: 
            # chance for double play at 0 or 1 outs 
            if outs < 2: 
                double_play = check_double_play()
                if double_play == True: 
                    return (1,outs+2), 0
                else: 
                    # chance for runners to advance on out 
                    advance_on_out = check_advance_on_out()
                    if advance_on_out == True: 
                        return (3,outs+1), 0
                    else: 
                        return (2,outs+1), 0
            # inning always ends at two outs 
            return (1, 3), 0
        
        elif state[0] == 3: 
            # advancement only if less than two outs 
            if outs < 2: 
                advance_on_out = check_advance_on_out()
                if advance_on_out == True: 
                    return (4,outs+1), 0 
                else:
                    return (3,outs+1), 0
            else:
                return (3, 3), 0
        
        elif state[0] == 4: 
            # chance to score on out
            score_on_out = check_score_on_out()
            if score_on_out == True: 
                return (1,outs+1), 1
            else:
                return (4,outs+1), 0
            
        elif state[0] == 5: 
            if outs < 2: 
                double_play = check_double_play()
                if double_play == True: 
                    return (4,outs+2), 0
                else:
                    advance_on_out = check_advance_on_out()
                    if advance_on_out == True: 
                        return (6,outs+1), 0
                    return (5,outs+1), 0
            return (5,outs+1), 0
        
        elif state[0] == 6: 
            double_play = check_double_play()
            score_on_out = check_score_on_out()
            
            if score_on_out == True and double_play == True: 
                return (1,outs+2), 1
            elif score_on_out == True and double_play == False: 
                return (3,outs+1), 1
            elif score_on_out == False and double_play == True:
                return (4,outs+2), 0
            else:
                return (7,outs+1), 0
            
        elif state[0] == 7: 
            score_on_out = check_score_on_out()
            advance_on_out = check_advance_on_out()
            if score_on_out == True: 
                if advance_on_out == True: 
                    return (4,outs+1), 1
                return (3,outs+1), 1
            else:
                if advance_on_out == True: 
                    return (4,outs+1), 0
                return (7,outs+1), 0
            
        elif state[0] == 8: 
            double_play = check_double_play()
            score_on_out = check_score_on_out()
            
            if score_on_out == True and double_play == True: 
                return (4,outs+2), 1
            elif score_on_out == True and double_play == False: 
                return (7,outs+1), 1
            elif score_on_out == False and double_play == True:
                return (7,outs+2), 0
            else:
                return (7,outs+2), 0

def simulate_half_inning(before_bip_dicts, after_bip_dicts, batter_due, debug=False):
    """
    simulates half an inning of a baseball game 

    input 1: a list of 9 dicts
    each dict contains probabilities of outcomes prior to ball in play 

    input 2: a list of 9 dicts
    each dict contains probabilities of outcomes after ball is in play

    both dicts are from batter's POV, already adjusted for pitcher skill 

    input 3: batter due up 

    output 1: runs scored this inning
    output 2: the index of the batter due next 
    output 3: dict of counts of events in this inning
        hits 
        walks
        strike outs
        at bats
    """

    # initialize runs and hits scored 
    runs_this_inning = 0
    hits_this_inning = 0
    walks_this_inning = 0
    at_bats_this_inning = 0
    strike_outs_this_inning = 0

    # start with batter in index 0, inning in state 0  
    batter_up = batter_due 
    state = (1,0)

    # cycle through batters infinitely until 3 outs reached 
    while True: 

        # every iteration is a new at bat
        at_bats_this_inning += 1
        
        # retrieve this batter's prob distributions
        before_bip = before_bip_dicts[batter_up]
        after_bip = after_bip_dicts[batter_up]

        # sample from before bip
        before_bip_population = list(before_bip.keys())
        before_bip_weights = list(before_bip.values())
        before_bip_sample = random.choices(
            population=before_bip_population,
            weights=before_bip_weights,
            k=1
        )[0]

        # sample to determine action to take
        if before_bip_sample in out_of_play:
            action = before_bip_sample
            if action == "WALK%":
                walks_this_inning += 1
            if action == "SO%":
                strike_outs_this_inning += 1
            if action == "HR%":
                hits_this_inning += 1
        else: 
            # sample from after bip 
            after_bip_population = list(after_bip.keys())
            after_bip_weights = list(after_bip.values())
            after_bip_sample = random.choices(
                population=after_bip_population,
                weights=after_bip_weights,
                k=1
            )[0]
            action = after_bip_sample
            if action == "S%" or action == "D%" or action == "T%" or action == "HR%":
                hits_this_inning += 1
        
        # transition to new state and count how many runs this batter scored
        new_state, runs_this_play = transition_states(state, action)

        # add runs scored to total 
        runs_this_inning += runs_this_play

        # print a report
        if debug == True: 
            print("BATTER INDEX: ", batter_up)
            print("BEFORE BIP ACTION: ", before_bip_sample)
            print("FINAL ACTION: ",action)
            print("RUNS THIS ACTION: ",runs_this_play)
            print("CURRENT STATE: ", new_state)
            print("OUTS: ",new_state[1])

        # inning is over at three strikes
        if new_state[1] > 2: 
            if debug == True: 
                print("!!! THREE STRIKES, INNING IS OVER !!!")
            break 

        # get ready to move to next batter, update the state and batter index
        if batter_up == 8: 
            batter_up = 0
        else:
            batter_up += 1
        state = new_state

    if batter_up == 8: 
        next_batter = 0
    else:
        next_batter = batter_up + 1

    if debug == True: 
        print(" ")
        print(" === INNING REPORT === ")
        print(" HITS THIS INNING: ", hits_this_inning)
        print(" WALKS THIS INNING: ", walks_this_inning)
        print(" RUNS THIS INNING: ", runs_this_inning)
        print(" AT BATS THIS INNING: ", at_bats_this_inning)
        print(" ")

    return runs_this_inning, next_batter, {"hits":hits_this_inning,
                                           "walks":walks_this_inning,
                                           "runs":runs_this_inning,
                                           "strike_outs":strike_outs_this_inning,
                                           "abs":at_bats_this_inning}

def simulate_innings(before_bip_dicts, after_bip_dicts, inning_count, debug=False):
    """
    simulates several innings of a mlb game, from perspective of a single team (either top or bottom half)

    input 1: a list of 9 dicts
    each dict contains probabilities of outcomes prior to ball in play 

    input 2: a list of 9 dicts
    each dict contains probabilities of outcomes after ball is in play

    input 3: how many innings to simulate

    -----------------------------

    output 1: a dict where keys are count categories and values are lists of counts per inning
    for example, {"RUNS":[1,0,0,0,0,0,4,0,1]}
        
    """

    runs_per_inning = [] 
    current_inning = 0
    inning_breakdown = dict()
    runs_total = []
    sos_total = []
    hits_total = []

    # start with batter in index 0 
    batter_up = 0

    while inning_count > current_inning: 
        if debug == True: 
            print("===== CURRENT INNING: ", current_inning, " =====")
        # simulate half an inning
        runs_this_inning, batter_up, this_inning_info = simulate_half_inning(before_bip_dicts, after_bip_dicts, batter_up, debug)
        # append number of runs 
        runs_per_inning.append(runs_this_inning)

        # update current inning 
        current_inning += 1

        # update combined inning info
        runs_total.append(this_inning_info["runs"])
        sos_total.append(this_inning_info["strike_outs"])
        hits_total.append(this_inning_info["hits"]+this_inning_info["walks"])

        # add to inning breakdowns
        inning_breakdown["runs"] = runs_total
        inning_breakdown["strike_outs"] = sos_total
        inning_breakdown["hits"] = hits_total

    return runs_per_inning, inning_breakdown

def gather_batting_data(home_abb, away_abb, date, bat_proj, pitch_proj):
    """
    finds players in an mlb game and returns probabilities of batting events 
    --------------------
    input 1: abbreviation of home team
    input 2: abbreviation of away team 
    input 3: date, in MM-DD-YYYY format 
    --------------------
    output: 4 lists, each containing dicts with probabilities of batting events
    1: home before bip
    2: home after bip
    3: away before bip
    4: away after bip
    """

    # initialize lists for home and away batters
    home_before_bip = []
    home_after_bip = []
    away_before_bip = []
    away_after_bip = []

    # get batters and pitchers
    batter_lists, pitcher_lists = mlb_stat_collection.get_game_rosters((home_abb,away_abb),date)

    if not any(pitcher_lists): 
        print(f"incomplete pitching data for {away_abb} @ {home_abb}")
        return None, None, None, None

    home_batters, away_batters = batter_lists
    home_pitcher = pitcher_lists[0][0]
    away_pitcher = pitcher_lists[1][0]

    # return None if lists are empty 
    if not home_batters: 
        print(f"incomplete batting data for {home_abb}, {away_abb} @ {home_abb}")
        return None, None, None, None 
    if not away_batters: 
        print(f"incomplete batting data for {away_abb}, {away_abb} @ {home_abb}")
        return None, None, None, None

    print(f" getting pitching data from {pitch_proj}")
    print(f" getting batting data from {bat_proj}")

    # get list of pitcher stats, as percents 
    home_pitcher_before_bip_dict = mlb_stat_collection.get_player_stats(pitch_proj, home_pitcher)
    away_pitcher_before_bip_dict = mlb_stat_collection.get_player_stats(pitch_proj, away_pitcher)
    if not home_pitcher_before_bip_dict: 
        print("no data on home pitcher")
        return None, None, None, None 
    if not away_pitcher_before_bip_dict: 
        print("no data on home pitcher")
        return None, None, None, None 
    home_pitcher_before_bip = mlb_stat_adjust.counts_to_percents_dict(mlb_stat_adjust.filter_pitcher_stats(home_pitcher_before_bip_dict))
    away_pitcher_before_bip = mlb_stat_adjust.counts_to_percents_dict(mlb_stat_adjust.filter_pitcher_stats(away_pitcher_before_bip_dict))

    # get list of dicts for batting stats, before and after bip
    for home_batter in home_batters:
        raw_home_batter_stats = mlb_stat_collection.get_player_stats(bat_proj, home_batter)
        
        # adjust for park factor
        pf_adjusted_home_batter = mlb_stat_adjust.normalize_pf(raw_home_batter_stats,home_abb)
        if pf_adjusted_home_batter == None: 
            print(f"ERROR on {home_batter}")
            return None, None, None, None

        # filter into before and after bip
        home_batter_before_bip1, home_batter_after_bip1 = mlb_stat_adjust.filter_batter_stats(pf_adjusted_home_batter)
       
        # convert counts to percents 
        home_batter_before_bip2 = mlb_stat_adjust.counts_to_percents_dict(home_batter_before_bip1)
        home_batter_after_bip = mlb_stat_adjust.counts_to_percents_dict(home_batter_after_bip1)

        # adjust before bip for skill 
        home_batter_before_bip = mlb_stat_adjust.log_five_scale(home_batter_before_bip2,away_pitcher_before_bip)

        # append to list
        home_before_bip.append(home_batter_before_bip)
        home_after_bip.append(home_batter_after_bip)

    # repeat for away team
    for away_batter in away_batters:
        raw_away_batter_stats = mlb_stat_collection.get_player_stats(bat_proj, away_batter)
        
        # adjust for park factor
        pf_adjusted_away_batter = mlb_stat_adjust.normalize_pf(raw_away_batter_stats,away_abb)
        if pf_adjusted_away_batter == None: 
            print(f"ERROR on {away_batter}")
            return None, None, None, None
        
        # filter into before and after bip
        away_batter_before_bip1, away_batter_after_bip1 = mlb_stat_adjust.filter_batter_stats(pf_adjusted_away_batter)
       
        # convert counts to percents 
        away_batter_before_bip2 = mlb_stat_adjust.counts_to_percents_dict(away_batter_before_bip1)
        away_batter_after_bip = mlb_stat_adjust.counts_to_percents_dict(away_batter_after_bip1)
    
        # adjust before bip for skill 
        away_batter_before_bip = mlb_stat_adjust.log_five_scale(away_batter_before_bip2,home_pitcher_before_bip)

        # append to list
        away_before_bip.append(away_batter_before_bip)
        away_after_bip.append(away_batter_after_bip)

    return home_before_bip, home_after_bip, away_before_bip, away_after_bip

def simulate_mlb_game(home_team, away_team, date, bat_data, pitch_data, iterations=5000, innings=9, thresh=8.5, debug=False):
    """
    simulates the result of an mlb game in the first 5 innings
    --------------------
    input 1: abbreviation of home team
    input 2: abbreviation of away team 
    input 3: date, in MM-DD-YYYY format 
    --------------------
    output 1: dict of probabilities for certain events
    output 2: list of 6 lists
        list 1: average number of home runs, per inning 
        list 2: average number of home hits and walks, per inning
        list 3: average number of home strike outs, per inning 
        lists 4 to 6 are same as above, but for away team 
    """

    # gather lists of dicts for each team
    # aab = away after ball in play 
    hbb_nopf_raw, hab_nopf_raw, abb_nopf, aab_nopf = gather_batting_data(home_team, away_team, date, bat_data, pitch_data)

    # no probs if no data
    if hbb_nopf_raw == None or hab_nopf_raw == None or abb_nopf == None or aab_nopf == None: 
        return None, None

    # apply home field advantage 
    hbb_nopf = mlb_stat_adjust.apply_home_adv(hbb_nopf_raw)
    hab_nopf = mlb_stat_adjust.apply_home_adv(hab_nopf_raw)

    # apply park factor adjustment 
    hbb = mlb_stat_adjust.reapply_pf(hbb_nopf, home_team)
    hab = mlb_stat_adjust.reapply_pf(hab_nopf, home_team)
    abb = mlb_stat_adjust.reapply_pf(abb_nopf, home_team)
    aab = mlb_stat_adjust.reapply_pf(aab_nopf, home_team)
    
    # apply pitcher decay adjustment

    # get relief pitcher dicts
    # relief pitcher goes into effect at start of sixth inning
    # relief pitcher is average of pitchers available in bullpen
    
    # inning by inning breakdown
    home_runs_distribution = [[],[],[],[],[],[],[],[],[]]
    home_hits_distribution = [[],[],[],[],[],[],[],[],[]]
    home_sos_distribution = [[],[],[],[],[],[],[],[],[]]
    away_runs_distribution = [[],[],[],[],[],[],[],[],[]]
    away_hits_distribution = [[],[],[],[],[],[],[],[],[]]
    away_sos_distribution = [[],[],[],[],[],[],[],[],[]]

    # initialize some variables
    current_iteration = 0 
    home_victories = 0
    away_victories = 0
    above_thresh = 0
    rifi_count = 0
    above_thresh_5th = 0

    # total home and away scores, for histogram
    total_home_scores = []
    total_away_scores = []

    # simulate one game at a time
    while current_iteration < iterations: 

        home_box_score, home_innings_info = simulate_innings(hbb,hab,innings, debug=False)
        away_box_score, away_innings_info = simulate_innings(abb,aab,innings, debug=False)

        # bottom of 9th inning adjustment for shut outs 
        home_score_through_8 = sum(home_box_score[0:7])
        away_score_through_9 = sum(away_box_score[0:8])
        home_score_9th_raw = home_box_score[8]
        middle_of_9_deficit = home_score_through_8 - away_score_through_9
        # game ends at middle of 9th, remove score at bottom of 9th
        if middle_of_9_deficit > 0: 
            home_box_score.pop()
            home_box_score.append(0)
            shut_out = True
            walk_off = False
        
        # bottom of 9th inning adjustment for walk offs
        # game ends if home team scores more than the deficit
        elif home_score_9th_raw > abs(middle_of_9_deficit): 
            home_score_9th_real = round(abs(middle_of_9_deficit) + 1, 2)
            home_box_score.pop()
            home_box_score.append(home_score_9th_real)
            walk_off = True
            shut_out = False 
        # away team always wins if it's not shut out or walk off, no adjustments
        else: 
            shut_out = False
            walk_off = False

        # go into extra innings if teams are tied after 9
        if sum(home_box_score) == sum(away_box_score): 
            while True: 
                extra_home, home_innings_info_extra = simulate_innings(hbb,hab,1) 
                extra_away, away_innings_info_extra = simulate_innings(abb,aab,1)
                home_box_score.append(extra_home[0])
                away_box_score.append(extra_away[0])
                if sum(home_box_score) != sum(away_box_score): 
                    break
        
        # start at inning with index 0 
        current_inning = 0 
        # tally up all on base, strike outs, and runs per inning
        while current_inning < innings:

            away_runs_distribution[current_inning].append(away_innings_info["runs"][current_inning])

            if current_inning < 8: 
                home_runs_distribution[current_inning].append(home_innings_info["runs"][current_inning])
            elif current_inning == 8: 
                if shut_out == True: 
                    home_runs_distribution[current_inning].append(0)
                if walk_off == True: 
                    home_runs_distribution[current_inning].append(home_score_9th_real)
                else:
                    home_runs_distribution[current_inning].append(home_innings_info["runs"][current_inning])

            home_sos_distribution[current_inning].append(home_innings_info["strike_outs"][current_inning])
            away_sos_distribution[current_inning].append(away_innings_info["strike_outs"][current_inning])

            home_hits_distribution[current_inning].append(home_innings_info["hits"][current_inning])
            away_hits_distribution[current_inning].append(away_innings_info["hits"][current_inning])

            current_inning += 1

        # tally up if home team won, if runs were above threshhold 
        home_score_this_iteration = sum(home_box_score)
        away_score_this_iteration = sum(away_box_score)
        if home_score_this_iteration > away_score_this_iteration:
            home_victories += 1
        if home_score_this_iteration < away_score_this_iteration:
            away_victories += 1
        if home_score_this_iteration + away_score_this_iteration > thresh: 
            above_thresh += 1 

        total_home_scores.append(home_score_this_iteration)
        total_away_scores.append(away_score_this_iteration)

        # for RIFI 
        if home_box_score[0] + away_box_score[0] > 0: 
            rifi_count += 1
        # for 5th inning
        if sum(home_box_score[:5]) + sum(away_box_score[:5]) > 4.5:
            above_thresh_5th += 1

        current_iteration += 1

    # find percentages
    home_win_percent = round(home_victories/iterations,2)
    above_thresh_percent = round(above_thresh/iterations,2)

    # find RIFI perentage
    rifi_percent = round(rifi_count/iterations,2)

    # find end of 5th inning percentage
    above_thresh_5th_percent = round(above_thresh_5th/iterations,2)

    # average all counting stats to match iterations
    avg_home_runs = [round(sum(inner) / len(inner),2) for inner in home_runs_distribution]
    avg_away_runs = [round(sum(inner) / len(inner),2) for inner in away_runs_distribution]

    avg_home_sos = [round(sum(inner) / len(inner),2) for inner in home_sos_distribution]
    avg_away_sos = [round(sum(inner) / len(inner),2) for inner in away_sos_distribution]

    avg_home_hits = [round(sum(inner) / len(inner),2) for inner in home_hits_distribution]
    avg_away_hits = [round(sum(inner) / len(inner),2) for inner in away_hits_distribution]

    averaged_lists = {
        "home_runs":avg_home_runs,"home_hits":avg_home_hits,"home_strike_outs":avg_home_sos,
        "away_runs":avg_away_runs,"away_hits":avg_away_hits,"away_strike_outs":avg_away_sos
    }

    print(averaged_lists)

    # print results
    print(f" RESULTS FOR {away_team} @ {home_team}")
    print(f" HOME WIN %: {home_win_percent}")
    print(f" ABOVE {thresh} %: {above_thresh_percent}")
    print(f" ABOVE 4.5 %: {above_thresh_5th_percent}")
    print(f" RIFI %: {rifi_percent}")

    if debug == True: 
        inning = "_In1_"
        index = 0
        create_and_save_histogram(home_runs_distribution[index], home_team + "_runs" + inning + date + ".png")
        create_and_save_histogram(away_runs_distribution[index], away_team + "_runs" + inning + date + ".png")

        create_and_save_histogram(home_sos_distribution[index], home_team + "_sos" + inning + date + ".png")
        create_and_save_histogram(away_sos_distribution[index], away_team + "_sos" + inning + date + ".png")

        create_and_save_histogram(home_hits_distribution[index], home_team + "_hits" + inning + date + ".png")
        create_and_save_histogram(away_hits_distribution[index], away_team + "_hits" + inning + date + ".png")

    probs_dict = {
        "Home_Win_Prob":home_win_percent,
        "Over_8_Prob":above_thresh_percent,
        "Over_4_Prob":above_thresh_5th_percent,
        "RIFI_Prob":rifi_percent
    }

    return probs_dict, averaged_lists

if __name__ == "__main__":
    import sys
    home_team = sys.argv[2].upper()
    away_team = sys.argv[3].upper()
    date = sys.argv[1]

    if len(sys.argv) == 5: 
        if sys.argv[4] == "STE":
            bat_proj = "Steamer"   
        else:
            bat_proj = sys.argv[4].upper()
        pitch_proj = bat_proj
    else:
        if sys.argv[4] == "STE":
            bat_proj = "Steamer"
        else: 
            bat_proj = sys.argv[4].upper()
        if sys.argv[5] == "STE":
            pitch_proj = "Steamer"
        else:
            pitch_proj = sys.argv[5].upper()


    bat_proj_path = bat_proj + "_Batters_may29.csv"
    pitch_proj_path = pitch_proj + "_Pitchers_may29.csv"

    if len(sys.argv) == 5: 
        if sys.argv[4].upper() == "ZIPO":
            bat_proj_path = "ZIPS_Batters_may24.csv"
            pitch_proj_path = "ZIPS_Pitchers_may24.csv"
        if sys.argv[4].upper() == "ATCO":
            bat_proj_path = "ATC_Batters_2026.csv"
            pitch_proj_path = "ATC_Pitchers_2026.csv"         

    simulate_mlb_game(home_team, away_team, date, bat_proj_path, pitch_proj_path, iterations=5000, innings=9, thresh=8.5)

    """

    before_bip_dicts = []
    after_bip_dicts = []
    before_bip_dict = {
        "HR%":0.05,
        "SO%":0.25,
        "BIP%":0.50,
        "WALK%":0.2
    }
    after_bip_dict = {
        "OIP%":0.5,
        "S%":0.43,
        "D%":0.05,
        "T%":0.01
    }

    while len(before_bip_dicts) < 9:
        before_bip_dicts.append(before_bip_dict) 
        after_bip_dicts.append(after_bip_dict)

    """

