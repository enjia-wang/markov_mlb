"""
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


# ── Build one CSV row from a game result dict ─────────────────────────────────

def _build_row(display_date: str, game: dict) -> dict:
    home_score = game["home_score"]
    away_score = game["away_score"]

    if home_score is not None and away_score is not None:
        total    = home_score + away_score
        home_win = 1 if home_score > away_score else 0
    else:
        total    = None
        home_win = None

    return {
        "Date":        display_date,
        "Home":        game["home_team"],
        "Away":        game["away_team"],
        "Home_Score":  home_score if home_score is not None else "",
        "Away_Score":  away_score if away_score is not None else "",
        "Total_Score": total      if total      is not None else "",
        "Home_Win":    home_win   if home_win   is not None else "",
    }

# ── Statcast: first-5-innings results ────────────────────────────────────────

def _statcast_results_f5(savant_date: str) -> list[dict]:
    """
    Like _statcast_results but caps scores at the end of the 5th inning.

    Filters pitches to inning <= 5, then takes the last pitch of the 5th
    inning per game. The score at that point is the F5 score.

    A game is only included if it has at least one pitch recorded in inning 5
    (guarantees the 5th inning was actually played).
    """
    print(f"  [statcast F5] Fetching {savant_date}...")
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

    score_cols = {"post_home_score", "post_away_score"}
    id_cols    = {"game_pk", "home_team", "away_team", "inning"}
    missing = (score_cols | id_cols) - set(raw.columns)
    if missing:
        print(f"  [WARN] Statcast data missing columns: {missing}.")
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

    # Keep only pitches from innings 1-5
    first_five = raw[raw["inning"] <= 5].copy()

    sort_cols = [c for c in ("inning", "at_bat_number", "pitch_number")
                 if c in first_five.columns]
    if sort_cols:
        first_five = first_five.sort_values(["game_pk"] + sort_cols)

    # Only include games that have reached inning 5
    games_with_5 = first_five[first_five["inning"] == 5]["game_pk"].unique()
    if len(games_with_5) == 0:
        print(f"  [WARN] No games have completed 5 innings yet on {savant_date}.")
        return []

    first_five = first_five[first_five["game_pk"].isin(games_with_5)]
    last_of_5  = first_five.groupby("game_pk", as_index=False).last()

    results = []
    for _, row in last_of_5.iterrows():
        home_score = int(row["post_home_score"]) if pd.notna(row["post_home_score"]) else None
        away_score = int(row["post_away_score"]) if pd.notna(row["post_away_score"]) else None
        results.append({
            "home_team":  row["home_team"],
            "away_team":  row["away_team"],
            "home_score": home_score,
            "away_score": away_score,
            "source":     "statcast_f5",
        })
        if home_score is not None and away_score is not None:
            label = ("home leads" if home_score > away_score
                     else "away leads" if away_score > home_score
                     else "tied")
        else:
            label = "score unavailable"
        print(f"  [statcast F5] {row['away_team']} @ {row['home_team']}: "
              f"after 5 — {away_score}-{home_score} ({label})")

    return results

def _build_row_f5(display_date: str, game: dict) -> dict:
    """
    Build a CSV row for F5 results.
    Home_Win = 1 if home leads after 5, 0 if tied or trailing.
    """
    home_score = game["home_score"]
    away_score = game["away_score"]

    if home_score is not None and away_score is not None:
        total    = home_score + away_score
        # Tied after 5 counts as 0 (home did not win the F5 market)
        home_win = 1 if home_score > away_score else 0
    else:
        total    = None
        home_win = None

    return {
        "Date":        display_date,
        "Home":        game["home_team"],
        "Away":        game["away_team"],
        "Home_Score":  home_score if home_score is not None else "",
        "Away_Score":  away_score if away_score is not None else "",
        "Total_Score": total      if total      is not None else "",
        "Home_Win":    home_win   if home_win   is not None else "",
    }

# ── fill inning score ────────────────────────────────────────────────

def fill_matchup_results(
    csv_path: str,
    date_range: tuple[str, str],
) -> None:
    """
    Look up MLB game results for every day in date_range and write them to a CSV.

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
    print(f"\nfill_matchup_results: {len(dates)} day(s) from "
          f"{start_dt.strftime('%m/%d/%Y')} to {end_dt.strftime('%m/%d/%Y')}")

    # ── Load existing CSV ────────────────────────────────────────────────────
    try:
        existing = pd.read_csv(csv_path, dtype=str)
        for col in CSV_COLS:
            if col not in existing.columns:
                existing[col] = ""
    except FileNotFoundError:
        existing = pd.DataFrame(columns=CSV_COLS)

    # Build a set of (Date, Home, Away) already in the file for deduplication
    existing_keys: set[tuple[str, str, str]] = set()
    for _, row in existing.iterrows():
        d = str(row.get("Date", "")).strip()
        h = str(row.get("Home", "")).strip()
        a = str(row.get("Away", "")).strip()
        if d and h and a and d.lower() != "nan":
            existing_keys.add((d, h, a))

    new_rows: list[dict] = []

    # ── Process each date ────────────────────────────────────────────────────
    for game_date in dates:
        savant_date  = game_date.strftime("%Y-%m-%d")
        display_date = game_date.strftime("%-m/%-d/%Y")  # matches CSV format: 5/6/2026

        print(f"\n{'─'*50}")
        print(f"Date: {display_date}")

        # Try statcast first
        games = _statcast_results(savant_date)

        # Fall back to website if statcast returned nothing
        if not games:
            print(f"  Statcast empty — trying website fallback...")
            games = _website_matchups(savant_date)

        if not games:
            print(f"  No game data found for {display_date}.")
            continue

        for game in games:
            row = _build_row(display_date, game)
            key = (row["Date"], row["Home"], row["Away"])

            if key in existing_keys:
                print(f"  SKIP (already in CSV): {row['Away']} @ {row['Home']}")
                continue

            new_rows.append(row)
            existing_keys.add(key)

    # ── Append and save ──────────────────────────────────────────────────────
    if not new_rows:
        print("\nNo new rows to write.")
        return

    new_df = pd.DataFrame(new_rows, columns=CSV_COLS)
    result = pd.concat([existing, new_df], ignore_index=True)
    result.to_csv(csv_path, index=False)

    print(f"\n✓ Wrote {len(new_rows)} new row(s) to {csv_path}")
    print(new_df[["Date", "Home", "Away", "Home_Score", "Away_Score", "Home_Win"]].to_string(index=False))
