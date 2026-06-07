"""
fill_matchup_results.py
-----------------------
Scrape MLB game results for a date range and write them to a CSV.

Public API
----------
fill_matchup_results(csv_path, date_range)

    csv_path   : path to write (appended to if it already exists; created if not)
    date_range : (start_date, end_date) as "MM-DD-YYYY" strings (inclusive)

CSV format
----------
Date, Home, Away, Home_Score, Away_Score, Total_Score, Home_Win

    Home_Win : 1 if the home team won, 0 if they lost, blank if the game
               was not completed or the score is unavailable.

Strategy
--------
For each date in the range:
  1. Call pybaseball.statcast() for that single day (no team filter).
     Extract game-level results from the post_home_score / post_away_score
     columns: group by game_pk, sort by inning + at_bat_number + pitch_number,
     and read the final score from the last pitch of each game.
  2. If statcast returns nothing (game not yet played, or data not yet ingested),
     still record the matchup (home/away teams) with blank score columns by
     scraping the MLB starting-lineups website for team names.

Existing rows in the CSV whose Date+Home+Away match a game already found are
skipped (not overwritten).
"""

from datetime import datetime, timedelta

import pandas as pd
import pybaseball
import requests
from bs4 import BeautifulSoup


# ── Constants ────────────────────────────────────────────────────────────────

_LINEUP_URL = "https://www.mlb.com/starting-lineups/{date}"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

CSV_COLS = ["Date", "Home", "Away", "Home_Score", "Away_Score",
            "Total_Score", "Home_Win"]


# ── Date helpers ─────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> datetime:
    """Accept MM-DD-YYYY, MM/DD/YYYY, MM-DD-YY, or MM/DD/YY."""
    normalised = date_str.replace("-", "/")
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(normalised, fmt)
        except ValueError:
            continue
    # pandas fallback
    return pd.to_datetime(normalised, dayfirst=False).to_pydatetime()


def _date_range(start: datetime, end: datetime) -> list[datetime]:
    """Return every date from start to end, inclusive."""
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)
    return dates


# ── Statcast: extract game results for one day ────────────────────────────────

def _statcast_results(savant_date: str) -> list[dict]:
    """
    Pull statcast pitch data for a single date and extract one result row
    per game. Returns a list of dicts with keys:
        home_team, away_team, home_score, away_score, source="statcast"

    Uses post_home_score / post_away_score from the last pitch of each game
    (sorted by inning, at_bat_number, pitch_number).
    """
    print(f"  [statcast] Fetching {savant_date}...")
    try:
        raw = pybaseball.statcast(
            start_dt=savant_date,
            end_dt=savant_date,
            verbose=False,
        )
    except Exception as exc:
        print(f"  [WARN] Statcast exception for {savant_date}: {exc}")
        return []

    if raw is None or raw.empty:
        print(f"  [WARN] Statcast returned no data for {savant_date}.")
        return []

    # Normalise team codes
    raw = raw.copy()
    for col in ("home_team", "away_team"):
        if col in raw.columns:
            raw[col] = raw[col].str.upper().str.strip()

    # Confirm required columns are present
    score_cols = {"post_home_score", "post_away_score"}
    id_cols    = {"game_pk", "home_team", "away_team"}
    missing = (score_cols | id_cols) - set(raw.columns)
    if missing:
        print(f"  [WARN] Statcast data missing columns: {missing}. "
              f"Available: {raw.columns.tolist()}")
        # If we at least have game identity, record matchups without scores
        if id_cols <= set(raw.columns):
            games = raw.groupby("game_pk", as_index=False)[
                ["home_team", "away_team"]
            ].first()
            return [
                {
                    "home_team":  r["home_team"],
                    "away_team":  r["away_team"],
                    "home_score": None,
                    "away_score": None,
                    "source":     "statcast_no_scores",
                }
                for _, r in games.iterrows()
            ]
        return []

    # Sort each game's pitches chronologically and take the last row = final score
    sort_cols = [c for c in ("inning", "at_bat_number", "pitch_number")
                 if c in raw.columns]
    if sort_cols:
        raw = raw.sort_values(["game_pk"] + sort_cols)
    else:
        print(f"  [WARN] Cannot sort pitches; scores may be inaccurate.")

    last_pitches = raw.groupby("game_pk", as_index=False).last()

    results = []
    for _, row in last_pitches.iterrows():
        home_score = int(row["post_home_score"]) if pd.notna(row["post_home_score"]) else None
        away_score = int(row["post_away_score"]) if pd.notna(row["post_away_score"]) else None
        results.append({
            "home_team":  row["home_team"],
            "away_team":  row["away_team"],
            "home_score": home_score,
            "away_score": away_score,
            "source":     "statcast",
        })
        print(f"  [statcast] {row['away_team']} @ {row['home_team']}: "
              f"{away_score}-{home_score} "
              f"({'home win' if home_score is not None and away_score is not None and home_score > away_score else 'away win' if home_score is not None and away_score is not None else 'score unavailable'})")

    return results


# ── Website fallback: matchup names only ─────────────────────────────────────

def _website_matchups(savant_date: str) -> list[dict]:
    """
    Scrape the MLB starting-lineups page to get home/away team abbreviations
    for all games on a date. Returns a list of dicts with home_team, away_team.
    Scores are left as None (game not yet played or results not posted).
    """
    url = _LINEUP_URL.format(date=savant_date)
    print(f"  [website] Fetching {url}...")
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [WARN] Could not fetch lineup page: {exc}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    for matchup_div in soup.find_all("div", class_="starting-lineups__matchup"):
        game_div = matchup_div.find("div", class_="starting-lineups__game")
        if not game_div:
            continue
        away_span = game_div.find("span", class_="starting-lineups__team-name--away")
        home_span = game_div.find("span", class_="starting-lineups__team-name--home")
        if not away_span or not home_span:
            continue
        away_link = away_span.find("a", class_="starting-lineups__team-name--link")
        home_link = home_span.find("a", class_="starting-lineups__team-name--link")
        if not away_link or not home_link:
            continue
        away = away_link.get("data-tri-code", "").strip().upper()
        home = home_link.get("data-tri-code", "").strip().upper()
        if away and home:
            results.append({
                "home_team":  home,
                "away_team":  away,
                "home_score": None,
                "away_score": None,
                "source":     "website",
            })
            print(f"  [website] {away} @ {home} (scores not yet available)")

    return results



# ── CSV column definitions ────────────────────────────────────────────────────

CSV_COLS = [
    "Date", "Home", "Away",
    # Full-game
    "Home_Score", "Away_Score", "Total_Score", "Home_Win",
    # First 5 innings
    "F5_Home_Score", "F5_Away_Score", "F5_Total_Score", "F5_Home_Win",
    # First inning
    "F1_Home_Score", "F1_Away_Score", "F1_Total_Score", "Any_Runs",
]


# ── Statcast: fetch once, extract full / F5 / F1 scores ──────────────────────

def _statcast_all_splits(savant_date: str) -> list[dict]:
    """
    Fetch statcast pitch data once for a date and compute full-game, F5, and F1
    scores for every game in a single pass.

    Returns a list of dicts with keys:
        home_team, away_team,
        home_score, away_score,            (full game)
        f5_home_score, f5_away_score,      (after 5 innings; None if not reached)
        f1_home_score, f1_away_score,      (after 1 inning; None if not reached)
        source
    """
    print(f"  [statcast] Fetching {savant_date}...")
    try:
        raw = pybaseball.statcast(start_dt=savant_date, end_dt=savant_date, verbose=False)
    except Exception as exc:
        print(f"  [WARN] Statcast exception for {savant_date}: {exc}")
        return []

    if raw is None or raw.empty:
        print(f"  [WARN] Statcast returned no data for {savant_date}.")
        return []

    raw = raw.copy()
    for col in ("home_team", "away_team"):
        if col in raw.columns:
            raw[col] = raw[col].str.upper().str.strip()

    id_cols    = {"game_pk", "home_team", "away_team"}
    score_cols = {"post_home_score", "post_away_score"}
    missing = (id_cols | score_cols) - set(raw.columns)
    if missing:
        print(f"  [WARN] Statcast data missing columns: {missing}.")
        if id_cols <= set(raw.columns):
            return [
                {"home_team": r["home_team"], "away_team": r["away_team"],
                 "home_score": None, "away_score": None,
                 "f5_home_score": None, "f5_away_score": None,
                 "f1_home_score": None, "f1_away_score": None,
                 "source": "statcast_no_scores"}
                for _, r in raw.groupby("game_pk", as_index=False)[
                    ["home_team","away_team"]].first().iterrows()
            ]
        return []

    sort_cols = [c for c in ("inning", "at_bat_number", "pitch_number") if c in raw.columns]
    if sort_cols:
        raw = raw.sort_values(["game_pk"] + sort_cols)

    results = []
    for game_pk, gdf in raw.groupby("game_pk"):
        home = gdf["home_team"].iloc[0]
        away = gdf["away_team"].iloc[0]

        def _last_score(df):
            s = df[["post_home_score","post_away_score"]].dropna()
            if s.empty:
                return None, None
            return int(s["post_home_score"].iloc[-1]), int(s["post_away_score"].iloc[-1])

        # Full game
        h_full, a_full = _last_score(gdf)

        # F5: only if inning 5 exists
        gdf5 = gdf[gdf["inning"] <= 5]
        if not gdf5.empty and (gdf5["inning"] == 5).any():
            h_f5, a_f5 = _last_score(gdf5)
        else:
            h_f5, a_f5 = None, None

        # F1: only if inning 1 exists
        gdf1 = gdf[gdf["inning"] == 1]
        if not gdf1.empty:
            h_f1, a_f1 = _last_score(gdf1)
        else:
            h_f1, a_f1 = None, None

        # Console summary
        full_lbl = f"{a_full}-{h_full}" if h_full is not None else "TBD"
        f5_lbl   = f"{a_f5}-{h_f5}"    if h_f5  is not None else "TBD"
        f1_lbl   = f"{a_f1}-{h_f1}"    if h_f1  is not None else "TBD"
        print(f"  [statcast] {away} @ {home}: "
              f"final={full_lbl}  F5={f5_lbl}  F1={f1_lbl}")

        results.append({
            "home_team": home, "away_team": away,
            "home_score": h_full, "away_score": a_full,
            "f5_home_score": h_f5, "f5_away_score": a_f5,
            "f1_home_score": h_f1, "f1_away_score": a_f1,
            "source": "statcast",
        })

    return results


# ── Build merged CSV row ──────────────────────────────────────────────────────

def _build_row(display_date: str, game: dict) -> dict:
    """Build one CSV row covering full-game, F5, and F1 columns."""

    def _scores(h, a):
        if h is not None and a is not None:
            return h, a, h + a
        return None, None, None

    # Full game
    h, a, tot = _scores(game["home_score"], game["away_score"])
    home_win   = (1 if h > a else 0) if h is not None else None

    # F5 — tie counts as 0 (home did not win F5 market)
    h5, a5, tot5 = _scores(game["f5_home_score"], game["f5_away_score"])
    f5_win        = (1 if h5 > a5 else 0) if h5 is not None else None

    # F1
    h1, a1, tot1 = _scores(game["f1_home_score"], game["f1_away_score"])
    any_runs      = (1 if tot1 > 0 else 0) if tot1 is not None else None

    def _v(x): return x if x is not None else ""

    return {
        "Date":          display_date,
        "Home":          game["home_team"],
        "Away":          game["away_team"],
        "Home_Score":    _v(h),
        "Away_Score":    _v(a),
        "Total_Score":   _v(tot),
        "Home_Win":      _v(home_win),
        "F5_Home_Score": _v(h5),
        "F5_Away_Score": _v(a5),
        "F5_Total_Score":_v(tot5),
        "F5_Home_Win":   _v(f5_win),
        "F1_Home_Score": _v(h1),
        "F1_Away_Score": _v(a1),
        "F1_Total_Score":_v(tot1),
        "Any_Runs":      _v(any_runs),
    }


# ── Main public function ──────────────────────────────────────────────────────

def fill_matchup_results(
    csv_path: str,
    date_range: tuple[str, str],
) -> None:
    """
    Look up MLB game results for every day in date_range and write them to a CSV.

    Fills full-game, first-5-inning (F5), and first-inning (F1) scores in a
    single statcast fetch per day. Column layout:

        Date, Home, Away,
        Home_Score, Away_Score, Total_Score, Home_Win,
        F5_Home_Score, F5_Away_Score, F5_Total_Score, F5_Home_Win,
        F1_Home_Score, F1_Away_Score, F1_Total_Score, Any_Runs

    Doubleheader handling
    ----------------------
    If statcast returns two games between the same home/away pair on the same
    date, both are skipped entirely to avoid ambiguity.

    Existing-row handling
    ----------------------
    If a (Date, Home, Away) row already exists in the CSV:
      - Missing stat columns are filled in from the new data.
      - Columns that are already populated are left unchanged.
      - No duplicate row is appended.
    If all stat columns are already filled, nothing is written.

    Parameters
    ----------
    csv_path   : path to the output CSV (created or appended to)
    date_range : (start_date, end_date) as "MM-DD-YYYY" strings, inclusive
    """
    _STAT_COLS = [c for c in CSV_COLS if c not in ("Date", "Home", "Away")]

    start_dt = _parse_date(date_range[0])
    end_dt   = _parse_date(date_range[1])

    if end_dt < start_dt:
        print("[ERROR] end_date is before start_date.")
        return

    dates = _date_range(start_dt, end_dt)
    print(f"\nfill_matchup_results: {len(dates)} day(s) from "
          f"{start_dt.strftime('%m/%d/%Y')} to {end_dt.strftime('%m/%d/%Y')}")

    # ── Load or create CSV ───────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path, dtype=str)
        for col in CSV_COLS:
            if col not in df.columns:
                df[col] = ""
    except FileNotFoundError:
        df = pd.DataFrame(columns=CSV_COLS)

    # Build lookup: (date_str, home, away) -> list of row indices
    existing_idx: dict[tuple[str,str,str], list[int]] = {}
    for idx, row in df.iterrows():
        d = str(row.get("Date", "")).strip()
        h = str(row.get("Home", "")).strip()
        a = str(row.get("Away", "")).strip()
        if d and h and a and d.lower() != "nan":
            key = (d, h, a)
            existing_idx.setdefault(key, []).append(idx)

    new_rows: list[dict] = []
    modified = False

    for game_date in dates:
        savant_date  = game_date.strftime("%Y-%m-%d")
        display_date = game_date.strftime("%-m/%-d/%Y")

        print(f"\n{'─'*50}")
        print(f"Date: {display_date}")

        games = _statcast_all_splits(savant_date)

        if not games:
            print(f"  Statcast empty — trying website fallback...")
            raw_matchups = _website_matchups(savant_date)
            games = [
                {**m, "f5_home_score": None, "f5_away_score": None,
                       "f1_home_score": None, "f1_away_score": None}
                for m in raw_matchups
            ]

        if not games:
            print(f"  No game data found for {display_date}.")
            continue

        # ── Detect doubleheaders within today's results ──────────────────────
        matchup_counts: dict[tuple[str,str], int] = {}
        for game in games:
            mk = (game["home_team"], game["away_team"])
            matchup_counts[mk] = matchup_counts.get(mk, 0) + 1

        for game in games:
            home = game["home_team"]
            away = game["away_team"]
            mk   = (home, away)

            if matchup_counts[mk] > 1:
                print(f"  SKIP doubleheader: {away} @ {home} "
                      f"({matchup_counts[mk]} games today — ambiguous)")
                continue

            row_data = _build_row(display_date, game)
            key = (display_date, home, away)

            if key in existing_idx:
                row_indices = existing_idx[key]

                if len(row_indices) > 1:
                    print(f"  SKIP (duplicate rows in CSV): {away} @ {home} on {display_date}")
                    continue

                existing_row_idx = row_indices[0]
                existing_row = df.loc[existing_row_idx]

                # Find which stat columns are missing (blank or NaN)
                missing_cols = [
                    col for col in _STAT_COLS
                    if str(existing_row.get(col, "")).strip() in ("", "nan")
                    and row_data.get(col, "") != ""
                ]

                if not missing_cols:
                    print(f"  SKIP (fully populated): {away} @ {home} on {display_date}")
                    continue

                # Fill missing columns in-place
                for col in missing_cols:
                    df.at[existing_row_idx, col] = row_data[col]

                print(f"  UPDATE {away} @ {home}: filled {missing_cols}")
                modified = True

            else:
                new_rows.append(row_data)
                existing_idx[key] = [-1]   # sentinel to catch future dupes this run

    # ── Append new rows ──────────────────────────────────────────────────────
    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=CSV_COLS)
        df = pd.concat([df, new_df], ignore_index=True)
        modified = True

    if not modified:
        print("\nNo changes to write.")
        return

    df.to_csv(csv_path, index=False)
    added = len(new_rows)
    print(f"\n✓ Saved to {csv_path}  "
          f"({added} row(s) added, {len(df) - added} existing row(s) potentially updated)")

# ── Inning-by-inning breakdown ────────────────────────────────────────────────

_OB_EVENTS = frozenset({
    "single", "double", "triple", "home_run",
    "walk", "intent_walk", "hit_by_pitch",
})
_SO_EVENTS = frozenset({
    "strikeout", "strikeout_double_play",
})

# All 54 stat columns: Home then Away, innings 1-9, stats Runs/OB/SO
_INNING_STAT_COLS = [
    f"{side}_{stat}_In{inn}"
    for side in ("Home", "Away")
    for inn in range(1, 10)
    for stat in ("Runs", "OB", "SO")
]

CSV_COLS_INNING = ["Date", "Home", "Away"] + _INNING_STAT_COLS


def _extract_inning_stats(game_df: pd.DataFrame) -> dict:
    """
    Given all statcast rows for a single game, return a dict of all 54
    inning stat values keyed by column name (e.g. 'Home_Runs_In1').

    Stat logic
    ----------
    Rows in 'Top' of inning N  → away team is batting
        Away_Runs_InN  : delta in post_away_score over those rows
        Away_OB_InN    : count of OB events in 'events' column
        Away_SO_InN    : count of SO events in 'events' column

    Rows in 'Bot' of inning N  → home team is batting
        Home_Runs_InN  : delta in post_home_score over those rows
        Home_OB_InN    : count of OB events
        Home_SO_InN    : count of SO events

    Innings with no rows get 0 for all three stats.
    Extra innings (inning > 9) are ignored.
    """
    game_df = game_df.copy()
    game_df["inning_topbot_norm"] = game_df["inning_topbot"].str.strip().str.title()

    sort_cols = [c for c in ("inning", "at_bat_number", "pitch_number")
                 if c in game_df.columns]
    if sort_cols:
        game_df = game_df.sort_values(sort_cols)

    result = {}

    for inn in range(1, 10):
        for half, side, score_col in (
            ("Top", "Away", "post_away_score"),   # away bats in Top
            ("Bot", "Home", "post_home_score"),   # home bats in Bot
        ):
            subset = game_df[
                (game_df["inning"] == inn) &
                (game_df["inning_topbot_norm"] == half)
            ]

            if subset.empty:
                runs = ob = so = 0
            else:
                # Runs: score at end of this half minus score just before this half began.
                #
                # Inning order within each inning: Top comes first, Bot comes second.
                # So for Top of inning N:  prior = all rows in innings 1..N-1
                # For Bot of inning N:     prior = all rows in innings 1..N-1
                #                                  + Top of inning N
                if score_col in subset.columns:
                    scores = subset[score_col].dropna()
                    if not scores.empty:
                        end_score = int(scores.iloc[-1])

                        if half == "Top":
                            # Top is always the first half — prior is strictly previous innings
                            prior_mask = game_df["inning"] < inn
                        else:
                            # Bot is second — prior includes same-inning Top
                            prior_mask = (
                                (game_df["inning"] < inn) |
                                (
                                    (game_df["inning"] == inn) &
                                    (game_df["inning_topbot_norm"] == "Top")
                                )
                            )

                        prior = game_df[prior_mask]
                        if score_col in prior.columns and not prior.empty:
                            prior_scores = prior[score_col].dropna()
                            start_score = int(prior_scores.iloc[-1]) if not prior_scores.empty else 0
                        else:
                            start_score = 0

                        runs = max(0, end_score - start_score)
                    else:
                        runs = 0
                else:
                    runs = 0

                # OB and SO: counted from 'events' column (one entry per at-bat)
                if "events" in subset.columns:
                    events = subset["events"].dropna().str.lower()
                    ob = int(events.isin(_OB_EVENTS).sum())
                    so = int(events.isin(_SO_EVENTS).sum())
                else:
                    ob = so = 0

            result[f"{side}_Runs_In{inn}"] = runs
            result[f"{side}_OB_In{inn}"]   = ob
            result[f"{side}_SO_In{inn}"]   = so

    return result


def _statcast_inning_breakdown(savant_date: str) -> list[dict]:
    """
    Pull statcast data for a single date and compute per-inning stats
    for every game (innings 1-9 only; extra innings ignored).

    Returns a list of dicts, one per game, with keys:
        home_team, away_team, inning_stats (dict of 54 column values), source
    """
    print(f"  [statcast inning] Fetching {savant_date}...")
    try:
        raw = pybaseball.statcast(
            start_dt=savant_date,
            end_dt=savant_date,
            verbose=False,
        )
    except Exception as exc:
        print(f"  [WARN] Statcast exception for {savant_date}: {exc}")
        return []

    if raw is None or raw.empty:
        print(f"  [WARN] Statcast returned no data for {savant_date}.")
        return []

    raw = raw.copy()
    for col in ("home_team", "away_team"):
        if col in raw.columns:
            raw[col] = raw[col].str.upper().str.strip()

    required = {"game_pk", "home_team", "away_team", "inning", "inning_topbot"}
    missing  = required - set(raw.columns)
    if missing:
        print(f"  [WARN] Statcast missing required columns: {missing}.")
        return []

    # Ignore extra innings
    raw = raw[raw["inning"] <= 9]

    results = []
    for game_pk, game_df in raw.groupby("game_pk"):
        home = game_df["home_team"].iloc[0]
        away = game_df["away_team"].iloc[0]

        game_df_norm = game_df.copy()
        game_df_norm["inning_topbot_norm"] = (
            game_df_norm["inning_topbot"].str.strip().str.title()
        )

        # A game is complete if inning 9 Top (away batting) has occurred.
        # The Bot half of inning 9 is absent when the home team leads after
        # 8.5 innings (walk-off / shutout scenario) — this is a valid completed
        # game. We only skip games that have no inning-9 data at all.
        has_9_top = (
            (game_df_norm["inning"] == 9) &
            (game_df_norm["inning_topbot_norm"] == "Top")
        ).any()
        has_9_bot = (
            (game_df_norm["inning"] == 9) &
            (game_df_norm["inning_topbot_norm"] == "Bot")
        ).any()

        if not has_9_top:
            print(f"  [WARN] {away} @ {home} (pk={game_pk}): "
                  "9th inning Top not found — game may be live or incomplete. Skipping.")
            continue

        if not has_9_bot:
            print(f"  [INFO] {away} @ {home} (pk={game_pk}): "
                  "no Bot-9 (home team won without batting in 9th). "
                  "Writing 0s for Home_*_In9.")

        # Check for extra innings: if any inning > 9 pitches exist in the
        # FULL raw data (before we clipped to <=9), the game went to extras.
        # We still include it — we just cap at inning 9.
        inning_stats = _extract_inning_stats(game_df)

        # Summarise for console
        h_runs = sum(inning_stats.get(f"Home_Runs_In{i}", 0) for i in range(1, 10))
        a_runs = sum(inning_stats.get(f"Away_Runs_In{i}", 0) for i in range(1, 10))
        print(f"  [statcast inning] {away} @ {home}: "
              f"final {a_runs}-{h_runs} (9-inning cap)")

        results.append({
            "home_team":    home,
            "away_team":    away,
            "inning_stats": inning_stats,
            "source":       "statcast_inning",
        })

    return results


def fill_matchup_results_inning(
    csv_path: str,
    date_range: tuple[str, str],
) -> None:
    """
    Write a per-inning statistical breakdown for every completed MLB game
    in the date range.

    CSV columns
    -----------
    Date, Home, Away,
    Home_Runs_In1, Home_OB_In1, Home_SO_In1,  (innings 1-9)
    Home_Runs_In2, ...
    Away_Runs_In1, Away_OB_In1, Away_SO_In1,
    Away_Runs_In2, ...

    54 stat columns total (2 teams × 9 innings × 3 stats).

    Extra-inning games are included but stats are capped at inning 9.
    Games tied after 9 that go to extra innings are still recorded —
    extra-inning runs do NOT appear in any column.
    Games that have not yet completed 9 innings are skipped entirely
    (they will be missing from the CSV until the game is finished).

    Parameters
    ----------
    csv_path   : path to the output CSV (created or appended to)
    date_range : (start_date, end_date) as "MM-DD-YYYY" strings, inclusive
    """
    start_dt = _parse_date(date_range[0])
    end_dt   = _parse_date(date_range[1])

    if end_dt < start_dt:
        print("[ERROR] end_date is before start_date.")
        return

    dates = _date_range(start_dt, end_dt)
    print(f"\nfill_matchup_results_inning: {len(dates)} day(s) from "
          f"{start_dt.strftime('%m/%d/%Y')} to {end_dt.strftime('%m/%d/%Y')}")

    # ── Load existing CSV ────────────────────────────────────────────────────
    try:
        existing = pd.read_csv(csv_path, dtype=str)
        for col in CSV_COLS_INNING:
            if col not in existing.columns:
                existing[col] = ""
    except FileNotFoundError:
        existing = pd.DataFrame(columns=CSV_COLS_INNING)

    existing_keys: set[tuple[str, str, str]] = set()
    for _, row in existing.iterrows():
        d = str(row.get("Date", "")).strip()
        h = str(row.get("Home", "")).strip()
        a = str(row.get("Away", "")).strip()
        if d and h and a and d.lower() != "nan":
            existing_keys.add((d, h, a))

    new_rows: list[dict] = []

    for game_date in dates:
        savant_date  = game_date.strftime("%Y-%m-%d")
        display_date = game_date.strftime("%-m/%-d/%Y")

        print(f"\n{'─'*55}")
        print(f"Date: {display_date} (inning breakdown)")

        games = _statcast_inning_breakdown(savant_date)

        if not games:
            print(f"  No completed game data found for {display_date}.")
            continue

        for game in games:
            row_dict = {
                "Date": display_date,
                "Home": game["home_team"],
                "Away": game["away_team"],
            }
            # Fill all 54 stat columns
            for col in _INNING_STAT_COLS:
                row_dict[col] = game["inning_stats"].get(col, 0)

            key = (row_dict["Date"], row_dict["Home"], row_dict["Away"])
            if key in existing_keys:
                print(f"  SKIP (already in CSV): {row_dict['Away']} @ {row_dict['Home']}")
                continue

            new_rows.append(row_dict)
            existing_keys.add(key)

    if not new_rows:
        print("\nNo new rows to write.")
        return

    new_df = pd.DataFrame(new_rows, columns=CSV_COLS_INNING)
    result  = pd.concat([existing, new_df], ignore_index=True)
    result.to_csv(csv_path, index=False)

    print(f"\n✓ Wrote {len(new_rows)} new row(s) to {csv_path}")
    summary_cols = ["Date", "Home", "Away",
                    "Home_Runs_In1", "Away_Runs_In1",
                    "Home_Runs_In9", "Away_Runs_In9"]
    summary_cols = [c for c in summary_cols if c in new_df.columns]
    print(new_df[summary_cols].to_string(index=False))



# write_probs and write_inning_averages have moved to write_data_results.py

# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        csv_path   = sys.argv[1] if len(sys.argv) > 1 else "matchups.csv"
        start_date = sys.argv[2] if len(sys.argv) > 2 else "04-01-2025"
        end_date   = sys.argv[3] if len(sys.argv) > 3 else "04-03-2025"
        fill_matchup_results(csv_path, (start_date, end_date))
    else: 
        fill_matchup_results_inning("mlb_archive.csv",("05-25-2025","09-10-2025"))
        # fill_matchup_results_inning("mlb_archive.csv",("04-01-2026","05-25-2026"))