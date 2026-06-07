import pandas as pd
from datetime import datetime

# Mapping from simulate_mlb_game output keys to CSV column prefixes
_SIM_KEY_TO_COL_PATTERN = {
    "home_runs":        "Home_Runs_In{inn}",
    "home_strike_outs": "Home_SO_In{inn}",
    "home_hits":        "Home_OB_In{inn}",
    "away_runs":        "Away_Runs_In{inn}",
    "away_strike_outs": "Away_SO_In{inn}",
    "away_hits":        "Away_OB_In{inn}",
}
 
def write_probs(csv_path: str,bat_proj: str,pitch_proj: str) -> None:
    """
    Fill probability columns for every row in the CSV that has Date, Home,
    and Away populated but is missing probability values.
 
    Calls simulate_mlb_game(home_abb, away_abb, date_MM_DD_YYYY) for each row.
 
    The first output of simulate_mlb_game is a dict whose keys are column names
    and values are probabilities, e.g.:
        {
            "Home_Win_Prob": 0.5,
            "Over_8_Prob":   0.5,
            "Over_4_Prob":   0.5,
            "RIFI_Prob":     0.5,
        }
    Each key is written directly as a column heading in the CSV. Columns that
    don't yet exist in the CSV are created automatically.
 
    A row is skipped if:
      - Date, Home, or Away is missing
      - All columns present in the dict are already filled for this row
      - simulate_mlb_game returns None or raises an exception
      - The first output is not a dict
 
    The CSV date column may be MM/DD/YYYY or MM-DD-YYYY; both are accepted and
    converted to MM-DD-YYYY before being passed to simulate_mlb_game.
 
    Parameters
    ----------
    csv_path : path to the CSV file (updated in-place)
    """
    try:
        from markov_mlb import simulate_mlb_game
    except ImportError:
        import sys
        if "simulate_mlb_game" not in sys.modules:
            raise ImportError(
                "simulate_mlb_game could not be imported. "
                "Ensure simulate_mlb_game.py is on the Python path."
            )
        simulate_mlb_game = sys.modules["simulate_mlb_game"].simulate_mlb_game
 
    # ── Load CSV ─────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {csv_path}")
        return
 
    skipped_failed: list[tuple[int, str]] = []
 
    # ── Process each row ─────────────────────────────────────────────────────
    for idx, row in df.iterrows():
        date_raw = str(row.get("Date", "")).strip()
        home_val = str(row.get("Home", "")).strip()
        away_val = str(row.get("Away", "")).strip()
 
        if not date_raw or date_raw.lower() == "nan":
            continue
        if not home_val or home_val.lower() == "nan":
            continue
        if not away_val or away_val.lower() == "nan":
            continue
 
        # ── Normalise date MM/DD/YYYY → MM-DD-YYYY ───────────────────────────
        try:
            normalised = date_raw.replace("/", "-")
            parsed = None
            for fmt in ("%m-%d-%Y", "%m-%d-%y"):
                try:
                    parsed = datetime.strptime(normalised, fmt)
                    break
                except ValueError:
                    continue
            if parsed is None:
                parsed = pd.to_datetime(date_raw, dayfirst=False).to_pydatetime()
            date_for_sim = parsed.strftime("%m-%d-%Y")
        except Exception as exc:
            print(f"  Row {idx}: cannot parse date '{date_raw}' — skipping. ({exc})")
            skipped_failed.append((idx, f"date parse error: {exc}"))
            continue
 
        # ── Call simulate_mlb_game ────────────────────────────────────────────
        try:
            result = simulate_mlb_game(home_val, away_val, date_for_sim, bat_proj, pitch_proj)
        except Exception as exc:
            print(f"  Row {idx} ({away_val} @ {home_val}): simulate raised {exc} — skipping.")
            skipped_failed.append((idx, f"simulate exception: {exc}"))
            continue
 
        if result is None:
            print(f"  Row {idx} ({away_val} @ {home_val}): simulate returned None — skipping.")
            skipped_failed.append((idx, "simulate returned None"))
            continue
 
        # Extract the first element of the result (dict of probabilities)
        try:
            prob_dict = result[0] if isinstance(result, (tuple, list)) else result
        except (TypeError, IndexError) as exc:
            print(f"  Row {idx}: cannot read first output: {exc} — skipping.")
            skipped_failed.append((idx, f"output access error: {exc}"))
            continue
 
        if not isinstance(prob_dict, dict):
            print(f"  Row {idx}: first output is {type(prob_dict).__name__}, expected dict — skipping.")
            skipped_failed.append((idx, f"first output not a dict: {type(prob_dict).__name__}"))
            continue
 
        if not prob_dict:
            print(f"  Row {idx} ({away_val} @ {home_val}): prob dict is empty — skipping.")
            skipped_failed.append((idx, "empty prob dict"))
            continue
 
        # Ensure all dict keys exist as columns (add blanks for new columns)
        for col in prob_dict:
            if col not in df.columns:
                df[col] = ""
 
        # Skip if every prob column is already filled for this row
        row = df.loc[idx]   # re-read after possible column additions
        all_filled = all(
            str(row.get(col, "")).strip() not in ("", "nan")
            for col in prob_dict
        )
        if all_filled:
            print(f"  Row {idx} ({away_val} @ {home_val}, {date_raw}): already filled — skipping.")
            continue
 
        print(f"\n── Row {idx}: {away_val} @ {home_val} on {date_for_sim} ──")
 
        # Write each probability value to its column
        written = []
        for col, val in prob_dict.items():
            if val is not None:
                df.at[idx, col] = str(val)
                written.append(f"{col}={val}")
 
        print(f"  {',  '.join(written)}")
 
    # ── Save ─────────────────────────────────────────────────────────────────
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved to: {csv_path}")
 
    if skipped_failed:
        print(f"\n{'─' * 55}")
        print(f"SKIPPED — {len(skipped_failed)} row(s) with errors:")
        print(f"  {'Row':>5}  Reason")
        print(f"  {'─'*5}  {'─'*40}")
        for row_idx, reason in skipped_failed:
            print(f"  {row_idx:>5}  {reason}")
    else:
        print("No rows were skipped due to errors.")

def write_inning_averages(csv_path: str) -> None:
    """
    Fill per-inning average columns for every game row in the CSV using
    simulate_mlb_game(home_abb, away_abb, date_MM_DD_YYYY).
 
    simulate_mlb_game is expected to return a tuple:
        (win_prob_result, averages_dict)
 
    where averages_dict has keys:
        "home_runs", "home_strike_outs", "home_hits",
        "away_runs", "away_strike_outs", "away_hits"
 
    Each value is a list of 9 floats — one per inning (index 0 = inning 1).
 
    These are written to the CSV columns:
        Home_Runs_In1 … Home_Runs_In9
        Home_SO_In1   … Home_SO_In9
        Home_OB_In1   … Home_OB_In9   (hits map to OB columns)
        Away_Runs_In1 … Away_Runs_In9
        Away_SO_In1   … Away_SO_In9
        Away_OB_In1   … Away_OB_In9
 
    Rows are skipped if:
      - Date, Home, or Away is missing
      - All 54 average columns are already filled
      - simulate_mlb_game returns None or raises an exception
      - The averages dict is missing a key or its list has fewer than 9 values
 
    The date in the CSV may be MM/DD/YYYY (slashes); it is converted to
    MM-DD-YYYY (dashes) before being passed to simulate_mlb_game.
 
    Parameters
    ----------
    csv_path : path to the CSV file (updated in-place)
    """
    try:
        from markov_mlb import simulate_mlb_game
    except ImportError:
        import sys
        if "simulate_mlb_game" not in sys.modules:
            raise ImportError(
                "simulate_mlb_game could not be imported. "
                "Ensure simulate_mlb_game.py is on the Python path."
            )
        simulate_mlb_game = sys.modules["simulate_mlb_game"].simulate_mlb_game
 
    # ── Load CSV ─────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {csv_path}")
        return
 
    # Ensure all 54 average columns exist
    all_avg_cols = [
        pat.format(inn=inn)
        for pat in _SIM_KEY_TO_COL_PATTERN.values()
        for inn in range(1, 10)
    ]
    for col in all_avg_cols:
        if col not in df.columns:
            df[col] = ""
 
    skipped_failed: list[tuple[int, str]] = []
 
    # ── Process each row ─────────────────────────────────────────────────────
    for idx, row in df.iterrows():
        date_raw = str(row.get("Date", "")).strip()
        home_val = str(row.get("Home", "")).strip()
        away_val = str(row.get("Away", "")).strip()
 
        if not date_raw or date_raw.lower() == "nan":
            continue
        if not home_val or home_val.lower() == "nan":
            continue
        if not away_val or away_val.lower() == "nan":
            continue
 
        # Skip if every average column is already filled for this row
        already_filled = all(
            str(row.get(col, "")).strip() not in ("", "nan")
            for col in all_avg_cols
        )
        if already_filled:
            print(f"  Row {idx} ({away_val} @ {home_val}, {date_raw}): already filled — skipping.")
            continue
 
        # ── Normalise date MM/DD/YYYY → MM-DD-YYYY ───────────────────────────
        try:
            normalised = date_raw.replace("/", "-")
            parsed = None
            for fmt in ("%m-%d-%Y", "%m-%d-%y"):
                try:
                    parsed = datetime.strptime(normalised, fmt)
                    break
                except ValueError:
                    continue
            if parsed is None:
                parsed = pd.to_datetime(date_raw, dayfirst=False).to_pydatetime()
            date_for_sim = parsed.strftime("%m-%d-%Y")
        except Exception as exc:
            print(f"  Row {idx}: cannot parse date '{date_raw}' — skipping. ({exc})")
            skipped_failed.append((idx, f"date parse error: {exc}"))
            continue
 
        print(f"\n── Row {idx}: {away_val} @ {home_val} on {date_for_sim} ──")
 
        # ── Call simulate_mlb_game ────────────────────────────────────────────
        try:
            result = simulate_mlb_game(home_val, away_val, date_for_sim)
        except Exception as exc:
            print(f"  simulate_mlb_game raised an exception: {exc} — skipping.")
            skipped_failed.append((idx, f"simulate exception: {exc}"))
            continue
 
        if result is None:
            print(f"  simulate_mlb_game returned None — skipping.")
            skipped_failed.append((idx, "simulate returned None"))
            continue
 
        # Unpack: (output1, output2, averages_dict)  — averages is the third element
        try:
            *_, averages = result   # takes the last element regardless of tuple length
        except (TypeError, ValueError) as exc:
            print(f"  Cannot unpack simulate_mlb_game result: {exc} — skipping.")
            skipped_failed.append((idx, f"unpack error: {exc}"))
            continue
 
        if averages is None:
            print(f"  Averages dict is None — skipping.")
            skipped_failed.append((idx, "averages is None"))
            continue
 
        # ── Write inning averages ─────────────────────────────────────────────
        written = []
        any_written = False
 
        for sim_key, col_pattern in _SIM_KEY_TO_COL_PATTERN.items():
            inning_vals = averages.get(sim_key)
            if inning_vals is None:
                print(f"  [WARN] Key '{sim_key}' missing from averages dict — skipping key.")
                continue
            if len(inning_vals) < 9:
                print(f"  [WARN] Key '{sim_key}' has {len(inning_vals)} values (expected 9) — skipping key.")
                continue
 
            for inn in range(1, 10):
                col = col_pattern.format(inn=inn)
                val = inning_vals[inn - 1]   # index 0 = inning 1
                if val is not None:
                    df.at[idx, col] = str(round(float(val), 4))
                    any_written = True
 
            written.append(sim_key)
 
        if any_written:
            print(f"  Written keys: {written}")
        else:
            print(f"  No values written for this row.")
 
    # ── Save ─────────────────────────────────────────────────────────────────
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved to: {csv_path}")
 
    if skipped_failed:
        print(f"\n{'─' * 55}")
        print(f"SKIPPED — {len(skipped_failed)} row(s) with errors:")
        print(f"  {'Row':>5}  Reason")
        print(f"  {'─'*5}  {'─'*40}")
        for row_idx, reason in skipped_failed:
            print(f"  {row_idx:>5}  {reason}")
    else:
        print("No rows were skipped due to errors.")

def clean_and_save_csv(csvpath, output_path=None):
    """
    Reads a CSV, removes rows with any blank values, 
    and saves the cleaned data.
    """
    # Load the CSV
    df = pd.read_csv(csvpath)
    
    # Drop rows where any column contains NaN (blank)
    cleaned_df = df.dropna()
    
    # If no output path is provided, overwrite the original file
    target_path = output_path if output_path else csvpath
    
    # Save the cleaned DataFrame
    # index=False prevents pandas from adding an extra column for row numbers
    cleaned_df.to_csv(target_path, index=False)
    
    print(f"Cleaned file saved successfully to: {target_path}")
    return cleaned_df

if __name__ == "__main__":
    import sys
    bat_proj = "ATC_Batters_2025.csv"
    pitch_proj = "ATC_Pitchers_2025.csv"
    csv_path = sys.argv[1] + ".csv"
    functionality = sys.argv[2]
    if functionality == "probs":
        write_probs(csv_path,bat_proj,pitch_proj)
    elif functionality == "clean":
        clean_and_save_csv(csv_path)