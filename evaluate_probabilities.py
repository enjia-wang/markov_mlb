"""
evaluate_probabilities(csv_path, prob_cols)
---------------------------------------------------------------
Evaluates probability columns against actual game outcomes.

Arguments
---------
csv_path  : str
    Path to a CSV that contains at least:
      - result colmn 
      - one or more probability columns (values 0-1)

prob_cols : list of (col_name, threshold) tuples
    Each tuple pairs a column name with a threshold (0-1).
    A bet is triggered when abs(prob - 0.5) > threshold,
    i.e. when the probability is more than `threshold` away from 0.5.
    Direction: prob > 0.5 + threshold  -> bet HOME WIN
               prob < 0.5 - threshold  -> bet HOME LOSS

Reports printed
---------------
1. Overall betting record   - correct / incorrect / total bets and accuracy
2. Disagreement accuracy    - when columns disagreed on direction, which was right
3. Lone-signal accuracy     - when exactly one column crossed its threshold, which was right
4. Pairwise agreement       - for every pair of columns, accuracy when both agreed
"""

import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def _bet_direction(prob: float, threshold: float) -> str | None:
    """
    Return "WIN" if prob is above 0.5+threshold,
           "LOSS" if prob is below 0.5-threshold,
           None if within the threshold band (no bet).
    """
    if prob > 0.5 + threshold:
        return "WIN"
    if prob < 0.5 - threshold:
        return "LOSS"
    return None


def evaluate_probabilities(
    csv_path: str,
    prob_cols: list[tuple[str, float]],
    result_col: str,
) -> None:
    """
    Evaluate probability columns against actual game results.

    Parameters
    ----------
    csv_path        : path to the CSV file
    prob_cols       : list of (column_name, threshold) tuples
    """

    # -- Load ----------------------------------------------------------------
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {csv_path}")
        return

    if result_col not in df.columns:
        print(f"[ERROR] CSV does not contain a {result_col} column.")
        return

    missing = [col for col, _ in prob_cols if col not in df.columns]
    if missing:
        print(f"[ERROR] Columns not found in CSV: {missing}")
        return

    # -- Trackers ------------------------------------------------------------

    # Report 1: overall betting record
    overall_correct   = 0
    overall_incorrect = 0

    # Report 2: disagreement accuracy  {col: {"correct": int, "incorrect": int}}
    disagreement_record: dict[str, dict] = {
        col: {"correct": 0, "incorrect": 0} for col, _ in prob_cols
    }

    # Report 3: lone-signal accuracy   {col: {"correct": int, "incorrect": int}}
    lone_record: dict[str, dict] = {
        col: {"correct": 0, "incorrect": 0} for col, _ in prob_cols
    }

    # Report 4: pairwise agreement accuracy
    #   key: frozenset({col_a, col_b}) -> {"correct": int, "incorrect": int}
    col_names_ordered = [col for col, _ in prob_cols]
    pair_record: dict[frozenset, dict] = {
        frozenset(pair): {"correct": 0, "incorrect": 0}
        for pair in itertools.combinations(col_names_ordered, 2)
    }

    # -- Row loop ------------------------------------------------------------
    for _, row in df.iterrows():

        # Skip rows without a known result
        result_raw = str(row.get(result_col, "")).strip()
        if result_raw.lower() in ("", "nan"):
            continue
        try:
            actual = int(float(result_raw))   # 1 = home win, 0 = home loss
        except ValueError:
            continue
        if actual not in (0, 1):
            continue

        actual_direction = "WIN" if actual == 1 else "LOSS"

        # Compute each column's signal for this row
        signals: dict[str, str | None] = {}
        for col, threshold in prob_cols:
            prob_raw = str(row.get(col, "")).strip()
            if prob_raw.lower() in ("", "nan"):
                signals[col] = None
                continue
            try:
                prob = float(prob_raw)
            except ValueError:
                signals[col] = None
                continue
            signals[col] = _bet_direction(prob, threshold)

        active = {col: sig for col, sig in signals.items() if sig is not None}

        # -- Report 4: pairwise -- track every agreeing pair regardless of betting
        for pair in itertools.combinations(col_names_ordered, 2):
            col_a, col_b = pair
            sig_a = signals.get(col_a)
            sig_b = signals.get(col_b)
            # Both must have an active (above-threshold) signal and agree in direction
            if sig_a is not None and sig_b is not None and sig_a == sig_b:
                correct = sig_a == actual_direction
                key = "correct" if correct else "incorrect"
                pair_record[frozenset(pair)][key] += 1

        # -- No active signals -> no bet
        if not active:
            continue

        directions = set(active.values())

        # -- Conflicting signals -> no bet; still track disagreement accuracy
        if len(directions) > 1:
            for col, sig in active.items():
                correct = sig == actual_direction
                key = "correct" if correct else "incorrect"
                disagreement_record[col][key] += 1
            continue

        # -- All active signals agree --
        bet_direction = next(iter(directions))

        # -- Place the bet
        bet_correct = bet_direction == actual_direction

        if bet_correct:
            overall_correct += 1
        else:
            overall_incorrect += 1

        # -- Lone-signal tracking
        if len(active) == 1:
            lone_col = next(iter(active))
            key = "correct" if bet_correct else "incorrect"
            lone_record[lone_col][key] += 1

    # -- Report 1: Overall betting record ------------------------------------
    total_bets = overall_correct + overall_incorrect
    accuracy   = (overall_correct / total_bets * 100) if total_bets else 0.0

    print("=" * 62)
    print("REPORT 1 -- Overall Betting Record")
    print("=" * 62)
    print(f"  Total bets placed : {total_bets}")
    print(f"  Correct           : {overall_correct}")
    print(f"  Incorrect         : {overall_incorrect}")
    print(f"  Accuracy          : {accuracy:.1f}%")

    # -- Report 2: Disagreement accuracy -------------------------------------
    print()
    print("=" * 62)
    print("REPORT 2 -- Accuracy During Column Disagreements")
    print("(rows where signals pointed in opposite directions)")
    print("=" * 62)

    dis_rows = []
    for col, rec in disagreement_record.items():
        c, i = rec["correct"], rec["incorrect"]
        total = c + i
        acc   = (c / total * 100) if total else 0.0
        dis_rows.append((col, c, i, total, acc))

    dis_rows.sort(key=lambda x: (-x[4], -x[3]))

    if all(r[3] == 0 for r in dis_rows):
        print("  No disagreements recorded.")
    else:
        print(f"  {'Column':<30} {'Correct':>7} {'Wrong':>7} {'Total':>7} {'Acc%':>7}")
        print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for col, c, i, total, acc in dis_rows:
            if total == 0:
                continue
            marker = " <- best" if col == dis_rows[0][0] else ""
            print(f"  {col:<30} {c:>7} {i:>7} {total:>7} {acc:>6.1f}%{marker}")

    # -- Report 3: Lone-signal accuracy --------------------------------------
    print()
    print("=" * 62)
    print("REPORT 3 -- Accuracy as Sole Signal Above Threshold")
    print("(rows where only this column triggered a bet)")
    print("=" * 62)

    lone_rows = []
    for col, rec in lone_record.items():
        c, i = rec["correct"], rec["incorrect"]
        total = c + i
        acc   = (c / total * 100) if total else 0.0
        lone_rows.append((col, c, i, total, acc))

    lone_rows.sort(key=lambda x: (-x[4], -x[3]))

    if all(r[3] == 0 for r in lone_rows):
        print("  No lone-signal bets recorded.")
    else:
        print(f"  {'Column':<30} {'Correct':>7} {'Wrong':>7} {'Total':>7} {'Acc%':>7}")
        print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for col, c, i, total, acc in lone_rows:
            if total == 0:
                continue
            marker = " <- best" if col == lone_rows[0][0] else ""
            print(f"  {col:<30} {c:>7} {i:>7} {total:>7} {acc:>6.1f}%{marker}")

    # -- Report 4: Pairwise agreement accuracy --------------------------------
    print()
    print("=" * 62)
    print("REPORT 4 -- Accuracy When Two Columns Agreed")
    print("(both columns above their threshold and same direction)")
    print("=" * 62)

    pair_rows = []
    for pair_set, rec in pair_record.items():
        c, i = rec["correct"], rec["incorrect"]
        total = c + i
        acc   = (c / total * 100) if total else 0.0
        ordered = [col for col in col_names_ordered if col in pair_set]
        label = f"{ordered[0]}  +  {ordered[1]}"
        pair_rows.append((label, c, i, total, acc))

    pair_rows.sort(key=lambda x: (-x[4], -x[3]))

    if all(r[3] == 0 for r in pair_rows):
        print("  No pairwise agreements recorded.")
    else:
        print(f"  {'Pair':<44} {'Correct':>7} {'Wrong':>7} {'Total':>7} {'Acc%':>7}")
        print(f"  {'-'*44} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        best_acc = pair_rows[0][4]
        for label, c, i, total, acc in pair_rows:
            if total == 0:
                continue
            marker = " <- best" if acc == best_acc else ""
            print(f"  {label:<44} {c:>7} {i:>7} {total:>7} {acc:>6.1f}%{marker}")

    print("=" * 62)

def evaluate_bets(csvpath, probs_and_results, earnings, output_csvpath=None, kelly=False):
    """
    Simulates placing bets across multiple probability/result columns based on 
    explicit ranges, tracks compound or fixed cash flows, plots the timeline,
    and optionally exports a detailed CSV of every bet placed.
    """
    # Load the data
    df = pd.read_csv(csvpath)
    
    all_bets_list = []
    
    # Iterate through the dictionary of columns and their respective ranges
    for (prob_col, result_col), ranges in probs_and_results.items():
        # Drop rows where this specific probability is missing
        temp_df = df.dropna(subset=[prob_col]).copy()
        
        if temp_df.empty:
            continue
            
        # Create a boolean mask to filter rows based on the acceptable ranges
        bet_mask = pd.Series(False, index=temp_df.index)
        for lower, upper in ranges:
            bet_mask = bet_mask | ((temp_df[prob_col] >= lower) & (temp_df[prob_col] <= upper))
            
        # Filter for rows where a bet is placed
        bets_df = temp_df[bet_mask].copy()
        
        if bets_df.empty:
            continue
            
        # Standardize columns so we can cleanly concatenate all bets later
        bets_df['bet_prob'] = bets_df[prob_col]
        bets_df['bet_result'] = bets_df[result_col]
        # We use the probability column name (e.g., 'Home_Win') as the bet type
        bets_df['bet_type'] = prob_col 
        
        # Determine the predicted outcome (> 0.5 means bet on 1, <= 0.5 means bet on 0)
        bets_df['prediction'] = (bets_df['bet_prob'] > 0.5).astype(int)
        
        # Evaluate accuracy
        bets_df['is_correct'] = bets_df['prediction'] == bets_df['bet_result']
        
        all_bets_list.append(bets_df)
        
    if not all_bets_list:
        print("No bets were placed based on the given criteria.")
        return
        
    # Combine all bets into a single dataframe
    combined_bets = pd.concat(all_bets_list)
    
    # Sort by the original dataframe index to maintain the chronological order of the games
    combined_bets = combined_bets.sort_index().reset_index()
    
    # Apply betting logic chronologically across all mixed bets
    if not kelly:
        combined_bets['cash_flow'] = combined_bets['is_correct'].apply(lambda x: earnings if x else -1.0)
        combined_bets['cash_pool'] = combined_bets['cash_flow'].cumsum()
        start_cash = 0
    else:
        start_cash = 10
        combined_bets['fraction'] = (combined_bets['bet_prob'] - 0.5).abs()
        combined_bets['multiplier'] = np.where(
            combined_bets['is_correct'],
            1 + (combined_bets['fraction'] * earnings),
            1 - combined_bets['fraction']
        )
        combined_bets['cash_pool'] = start_cash * combined_bets['multiplier'].cumprod()
    
    # --- CSV Export Logic ---
    if output_csvpath:
        export_df = pd.DataFrame()
        # Use .get() to safely pull columns in case they don't exist in the original CSV
        export_df['Date'] = combined_bets.get('Date', pd.Series([np.nan]*len(combined_bets)))
        export_df['Home'] = combined_bets.get('Home', pd.Series([np.nan]*len(combined_bets)))
        export_df['Away'] = combined_bets.get('Away', pd.Series([np.nan]*len(combined_bets)))
        export_df['bet_type'] = combined_bets['bet_type']
        export_df['probability'] = combined_bets['bet_prob'] # Raw assigned probability
        export_df['result'] = combined_bets['is_correct'].astype(int) # 1 for win, 0 for loss
        
        export_df.to_csv(output_csvpath, index=False)
        print(f"Exported detailed log of {len(export_df)} bets to '{output_csvpath}'")

    # Generate the report
    total_bets = len(combined_bets)
    wins = combined_bets['is_correct'].sum()
    losses = total_bets - wins
    win_rate = (wins / total_bets) * 100
    final_cash = combined_bets['cash_pool'].iloc[-1]
    
    print("\n--- Betting Performance Report ---")
    print(f"Total Bets Placed: {total_bets}")
    print(f"Winning Bets:      {wins}")
    print(f"Losing Bets:       {losses}")
    print(f"Win Rate:          {win_rate:.2f}%")
    
    if kelly:
        print(f"Final Pool Size:   ${final_cash:.2f} (Start: $10)")
    else:
        print(f"Final Earnings:    {final_cash:+.2f} (Start: $0)")
        
    print("\n--- Breakdown by Bet Type ---")
    breakdown = combined_bets.groupby('bet_type')['is_correct'].agg(['count', 'mean'])
    breakdown['mean'] = (breakdown['mean'] * 100).round(2).astype(str) + '%'
    breakdown.columns = ['Total Bets', 'Win Rate']
    print(breakdown.to_string())
    
    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    
    # Plot the main cash flow line
    plt.plot(range(1, total_bets + 1), combined_bets['cash_pool'], color='blue', linewidth=1.5)
    
    # Identify indices where the Date changes
    if 'Date' in combined_bets.columns:
        date_changes = combined_bets['Date'] != combined_bets['Date'].shift(1)
        change_indices = combined_bets.index[date_changes].tolist()
        
        x_ticks = []
        x_labels = []
        
        for idx in change_indices:
            x_pos = idx + 1 
            date_label = combined_bets.loc[idx, 'Date']
            
            x_ticks.append(x_pos)
            x_labels.append(date_label)
            
            plt.axvline(x=x_pos, color='gray', linestyle='--', alpha=0.5)
            
        plt.axvline(x=total_bets, color='gray', linestyle='--', alpha=0.5)
        plt.xticks(x_ticks, x_labels, rotation=45, ha='right', fontsize=9)
    else:
        print("\nNote: 'Date' column not found. Defaulting to standard numerical x-axis.")
    
    # Add starting cash reference line
    plt.axhline(start_cash, color='red', linestyle='--', linewidth=1, label=f'Starting Pool (${start_cash})')
    
    # Formatting
    mode_title = ' (Kelly Fractional Betting)' if kelly else ' (Fixed Unit Betting)'
    plt.title(f'Cumulative Cash Flows Over Time{mode_title}')
    plt.xlabel('Date' if 'Date' in combined_bets.columns else 'Number of Bets Placed')
    plt.ylabel('Running Cash Pool')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3) 
    plt.tight_layout()
    
    plt.savefig('cash_flows.jpg', dpi=300)
    plt.close()
    print("\nPlot saved successfully as 'cash_flows.jpg'.")

def check_calibration(csvpath, prob_col, actual_col, bins):
    """
    Reads predictions and actuals from a CSV, bins the probabilities, 
    prints a calibration report, and returns a dictionary of the results.
    """
    
    # 1. Validate and expand bins to cover 0.0 and 1.0 if needed
    min_val = min([b[0] for b in bins])
    max_val = max([b[1] for b in bins])
    
    if min_val > 0:
        bins.append((0.0, min_val))
    if max_val < 1.0:
        bins.append((max_val, 1.0))
        
    # Sort bins to ensure a logical sequence from 0 to 1
    bins = sorted(bins, key=lambda x: x[0])
    
    # 2. Load the dataset
    df = pd.read_csv(csvpath)
    
    # 3. Helper function to map probabilities strictly to our tuple bins
    def assign_bin(val):
        for i, b in enumerate(bins):
            # First bin includes both lower and upper bounds [a, b]
            if i == 0:
                if b[0] <= val <= b[1]:
                    return b
            # Subsequent bins exclude the lower bound (a, b]
            else:
                if b[0] < val <= b[1]:
                    return b
        return np.nan 

    # Apply the mapping
    df['bin_tuple'] = df[prob_col].apply(assign_bin)
    
    # 4. Group by bins and calculate counts and proportions
    report_data = df.groupby('bin_tuple')[actual_col].agg(['count', 'mean'])
    
    # FIX: Convert the DataFrame to a native Python dictionary to avoid tuple-indexing errors
    report_dict = report_data.to_dict('index')
    
    # 5. Print the calibration report
    print(f"{'Bin':<20} | {'Count':<10} | {'Proportion of 1s'}")
    print("-" * 55)
    
    result_dict = {}
    
    for b in bins:
        # Now we check the native Python dictionary safely
        if b in report_dict:
            count = int(report_dict[b]['count'])
            prop = float(report_dict[b]['mean'])
            prop_str = f"{prop:.4f}"
            result_dict[b] = prop
        else:
            count = 0
            prop_str = "N/A"
            result_dict[b] = None 
            
        print(f"{str(b):<20} | {count:<10} | {prop_str}")
        
    return result_dict

def check_calibration(csvpath, prob_col, actual_col, bins, output_png=None):
    """
    Reads predictions and actuals from a CSV, bins the probabilities, 
    prints a calibration report, optionally saves a plot, and returns a dictionary.
    """
    
    # 1. Validate and expand bins to cover 0.0 and 1.0 if needed
    min_val = min([b[0] for b in bins])
    max_val = max([b[1] for b in bins])
    
    if min_val > 0:
        bins.append((0.0, min_val))
    if max_val < 1.0:
        bins.append((max_val, 1.0))
        
    # Sort bins to ensure a logical sequence from 0 to 1
    bins = sorted(bins, key=lambda x: x[0])
    
    # 2. Load the dataset
    df = pd.read_csv(csvpath)
    
    # 3. Print the header
    print(f"{'Bin':<20} | {'Count':<10} | {'Proportion of 1s'}")
    print("-" * 55)
    
    result_dict = {}
    
    # Variables to store coordinates and dynamic limits for our plot
    plot_x = []
    plot_y = []
    min_populated_bound = None
    max_populated_bound = None
    
    # 4. Iterate through bins and calculate directly
    for i, b in enumerate(bins):
        # Create a boolean mask for the current bin bounds
        if i == 0:
            mask = (df[prob_col] >= b[0]) & (df[prob_col] <= b[1])
        else:
            mask = (df[prob_col] > b[0]) & (df[prob_col] <= b[1])
            
        # Filter the DataFrame
        in_bin = df[mask]
        count = len(in_bin)
        
        # 5. Handle empty vs populated bins safely
        if count > 0:
            prop = float(in_bin[actual_col].mean())
            prop_str = f"{prop:.4f}"
            result_dict[b] = prop
            
            # Store the midpoint and actual proportion for plotting
            plot_x.append((b[0] + b[1]) / 2.0)
            plot_y.append(prop)
            
            # Track the lowest and highest bounds of populated bins
            if min_populated_bound is None:
                min_populated_bound = b[0]
            max_populated_bound = b[1] # Overwrites until the highest populated bin
            
        else:
            prop_str = "N/A"
            result_dict[b] = None
            
        print(f"{str(b):<20} | {count:<10} | {prop_str}")
        
    # 6. Plotting the Calibration Curve (Only if a filename is provided)
    if output_png and min_populated_bound is not None:
        
        # Plot the ideal "Perfectly Calibrated" line (y = x)
        plt.plot(
            [min_populated_bound, max_populated_bound], 
            [min_populated_bound, max_populated_bound], 
            'k--', 
            label="Perfectly Calibrated"
        )
        
        # Plot the actual bin results
        plt.plot(plot_x, plot_y, marker='o', linestyle='-', color='b', label="Model Calibration")
        
        # Calculate a 5% margin based on the populated range so edge points don't clip
        rng = max_populated_bound - min_populated_bound
        margin = rng * 0.05 if rng > 0 else 0.05
        
        # Formatting the plot dynamically
        plt.xlabel("Predicted Probability (Bin Midpoint)")
        plt.ylabel("Actual Proportion of 1s")
        plt.title("Calibration Plot")
        plt.xlim([min_populated_bound - margin, max_populated_bound + margin])
        plt.ylim([min_populated_bound - margin, max_populated_bound + margin])
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc="best")
        
        # Save the file and clear the plot buffer
        plt.savefig(output_png)
        plt.clf()
        
        print(f"\nCalibration plot saved successfully to: {output_png}")
    
    return result_dict

def create_bin_list(range_tuple, bin_width, precision=4):
    """
    Creates a list of bin tuples from a given range and bin width using pure Python.
    
    Args:
        range_tuple (tuple): (lower_bound, upper_bound)
        bin_width (float): The desired width for each bin
        precision (int): Number of decimal places to round to (default 4)
        
    Returns:
        list: A regular Python list of tuples, e.g., [(0.0, 0.2), (0.2, 0.4), ...]
    """
    start, end = range_tuple
    
    if bin_width <= 0:
        raise ValueError("Bin width must be a positive number.")
    if start >= end:
        raise ValueError("Lower bound must be less than upper bound.")

    bins = []
    current_lower = start
    
    while current_lower < end:
        # Calculate the next upper bound
        current_upper = current_lower + bin_width
        
        # If the next step overshoots the end, snap it to the end
        if current_upper > end or abs(end - current_upper) < 1e-9:
            current_upper = end
            
        # Round the values to fix floating-point precision issues
        bin_tuple = (round(current_lower, precision), round(current_upper, precision))
        bins.append(bin_tuple)
        
        # Move to the next bin
        current_lower = current_upper
        
    return bins

if __name__ == "__main__":

    import sys
    if sys.argv[1] == "cali":
        csvpath = sys.argv[2]
        bins = create_bin_list((0.2,0.8),0.025)
        print(" overall over/under ")
        check_calibration(csvpath, "Over_8_Prob", "Over_8", bins, "Over8Cali.png")
        print(" ")
        print(" overall win/loss ")
        check_calibration(csvpath, "Home_Win_Prob", "Home_Win", bins, "MoneylineCali.png")
        print(" ")
        print(" 5th inning over/under ")
        check_calibration(csvpath, "Over_4_Prob", "Over_4", bins, "Over4Cali.png")
        print(" ")
        print(" RIFI ")
        check_calibration(csvpath, "RIFI_Prob", "Any_Runs", bins, "RIFICali.png")

    elif sys.argv[1] == "bets":
        csvpath = sys.argv[2]
        bet_dict = {
            # ("Home_Win_Prob","Home_Win"):[(0,0.35),(0.6,0.65)],
            ("Home_Win_Prob","Home_Win"):[(0,0.375),(0.575,0.625),(0.65,1)],
            # ("Over_8_Prob","Over_8"):[(0.6,0.65),(0.7,1)],
            ("Over_8_Prob","Over_8"):[(0.6,0.65),(0.725,1)],
            # ("Over_4_Prob","Over_4"):[(0,0.4),(0.6,1)],
            ("Over_4_Prob","Over_4"):[(0,0.375),(0.625,1)],
            # ("RIFI_Prob","Any_Runs"):[(0.6,1)],
            ("RIFI_Prob","Any_Runs"):[(0,0.4),(0.6,1)]
        }

        """
        these are for zips 
        bet_dict = {
            ("Home_Win_Prob","Home_Win"):[(0,0.375),(0.575,0.7)],
            ("Over_8_Prob","Over_8"):[(0,0.4)],
            ("Over_4_Prob","Over_4"):[(0,0.35),(0.375,0.4),(0.6,0.725)],
            ("RIFI_Prob","Any_Runs"):[(0.6,1)]
        }


        """
        csv_output = "lastweek1.csv"
        evaluate_bets(csvpath,bet_dict,0.71,csv_output,False)