"""
mlb_roster_lookup.py
--------------------
Retrieve the starting lineup (batters + pitchers) for both teams in an MLB game.

Public API
----------
get_game_rosters(matchup, date)

    matchup : (home_abbr, away_abbr)  e.g. ("SD", "LAD")
    date    : MM-DD-YYYY string        e.g. "05-18-2026"

Returns
-------
batters  : ((home_batters_list, away_batters_list))
           Each list has 9 full-name strings in batting order.
pitchers : ((home_pitcher_list, away_pitcher_list))
           Each list has 1 full-name string (the starting pitcher).

Name formatting
---------------
Names are "First Last" with no extra transformation — the Chadwick register
stores first names like "J.P." already correctly, so joining name_first + " " +
name_last naturally produces "J.P. Crawford".  When falling back to the MLB
website, the visible link text is used directly (e.g. "J.P. Crawford").

Strategy
--------
1. Statcast (pitch-by-pitch data):
   - Batters:  group by batter MLBAM id + batting_order, split by inning half
               (Top = away batters, Bot = home batters), sort, resolve names.
   - Pitchers: inning-1 pitcher per side, sorted by at_bat_number / pitch_number.
2. MLB starting-lineups website fallback (if statcast has no/incomplete data):
   - Uses the same scraping logic as mlb_batting.py and mlb_pitcher_stats.py.
"""

import re
import sys
import os
from datetime import datetime

import pandas as pd
import pybaseball
import requests
from bs4 import BeautifulSoup
import unicodedata

# ── Shared constants ──────────────────────────────────────────────────────────

_LINEUP_BASE = "https://www.mlb.com/starting-lineups/{date}"
_LINEUP_DEFAULT = "https://www.mlb.com/starting-lineups/"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ── Date parsing ─────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> tuple[datetime, str]:
    """
    Parse MM-DD-YYYY (or MM/DD/YYYY, MM-DD-YY etc.) into a datetime and
    a YYYY-MM-DD string for statcast / URL building.
    """
    # Normalise separators so both '-' and '/' work
    normalised = date_str.replace("-", "/")
    game_date = None
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            game_date = datetime.strptime(normalised, fmt)
            break
        except ValueError:
            continue
    if game_date is None:
        # pandas inference as a last resort
        game_date = pd.to_datetime(normalised, dayfirst=False).to_pydatetime()
    return game_date, game_date.strftime("%Y-%m-%d")


# ── Name helpers ─────────────────────────────────────────────────────────────

def _chadwick_name(first: str, last: str) -> str:
    """
    Join first + last names from the Chadwick register without any case
    transformation.  The register already stores 'J.P.' correctly.
    """
    return f"{first.strip()} {last.strip()}"


def _slug_to_display_name(href: str) -> str:
    """
    Convert /player/j-p-crawford-572122 to a best-effort display name.
    Parts that are 1-2 chars are uppercased (initials), others capitalised.
    Dots are NOT inserted — use the visible link text instead when available.
    """
    slug = href.rstrip("/").split("/")[-1]
    parts = slug.split("-")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(p.upper() if len(p) <= 2 else p.capitalize() for p in parts)


def _clean_display_name(raw: str) -> str:
    """
    Clean a name string from the MLB website visible text.
    Collapses internal whitespace and strips edges.
    E.g. "J.P.  Crawford " -> "J.P. Crawford"
    """
    return re.sub(r"\s+", " ", raw.strip())


# ── Statcast: batter lineup ───────────────────────────────────────────────────

def _statcast_batters(
    savant_date: str,
    home_abbr: str,
    away_abbr: str,
) -> tuple[list[str], list[str]] | None:
    """
    Pull pitch-by-pitch data for the game and extract the batting orders.
    Returns (home_batters, away_batters) each ordered 1-9, or None on failure.

    Statcast inning convention:
        Top of inning → away team bats  → home pitcher pitches
        Bot of inning → home team bats  → away pitcher pitches
    So away batters appear in rows where inning_topbot is 'Top' (or 'top'),
    and home batters appear in 'Bot' / 'bot' rows.
    """
    try:
        # Fetch without team filter to get all pitches in both halves
        raw = pybaseball.statcast(
            start_dt=savant_date,
            end_dt=savant_date,
            verbose=False,
        )
    except Exception as exc:
        print(f"[WARN] Statcast fetch failed for {savant_date}: {exc}")
        return None

    if raw is None or raw.empty:
        print(f"[WARN] Statcast returned no data for {savant_date}.")
        return None

    required = {"batter", "batting_order", "home_team", "away_team", "inning_topbot"}
    missing = required - set(raw.columns)
    if missing:
        print(f"[WARN] Statcast data missing columns: {missing}")
        return None

    # Normalise team codes and inning half
    raw = raw.copy()
    raw["home_team"] = raw["home_team"].str.upper().str.strip()
    raw["away_team"] = raw["away_team"].str.upper().str.strip()
    raw["inning_topbot_norm"] = raw["inning_topbot"].str.strip().str.title()

    # Filter to the specific game by team codes
    game = raw[
        (raw["home_team"] == home_abbr.upper()) &
        (raw["away_team"] == away_abbr.upper())
    ]
    if game.empty:
        print(f"[WARN] Statcast: no rows found for {away_abbr} @ {home_abbr} on {savant_date}.")
        return None

    print(f"[DEBUG] Statcast: {len(game)} pitches for {away_abbr} @ {home_abbr}.")

    def _extract_ordered_batters(half: str) -> list[str]:
        """Extract batter MLBAM ids ordered by batting_order for one half-inning."""
        subset = game[game["inning_topbot_norm"] == half].copy()
        if subset.empty or "batting_order" not in subset.columns:
            return []
        # Drop rows with null batting_order, keep first appearance of each batter
        subset = subset.dropna(subset=["batting_order"])
        # Each batter has a consistent batting_order; take one row per batter
        unique = (
            subset.groupby("batter", as_index=False)["batting_order"]
            .first()
            .sort_values("batting_order")
        )
        return unique["batter"].astype(int).tolist()

    away_ids = _extract_ordered_batters("Top")   # away bats in Top
    home_ids = _extract_ordered_batters("Bot")   # home bats in Bot

    print(f"[DEBUG] Statcast batter IDs — home ({len(home_ids)}): {home_ids[:3]}... "
          f"away ({len(away_ids)}): {away_ids[:3]}...")

    if len(home_ids) < 9 or len(away_ids) < 9:
        print(f"[WARN] Statcast: incomplete lineup — home={len(home_ids)}, "
              f"away={len(away_ids)} batters. Falling back to website.")
        return None

    # Resolve MLBAM ids → names via Chadwick register
    all_ids = list(set(home_ids[:9] + away_ids[:9]))
    try:
        lookup = pybaseball.playerid_reverse_lookup(all_ids, key_type="mlbam")
    except Exception as exc:
        print(f"[WARN] playerid_reverse_lookup failed: {exc}")
        return None

    id_to_name = {
        int(row["key_mlbam"]): _chadwick_name(row["name_first"], row["name_last"])
        for _, row in lookup.iterrows()
        if pd.notna(row.get("key_mlbam"))
    }

    def _resolve(ids: list[int]) -> list[str]:
        names = []
        for mlbam_id in ids[:9]:
            name = id_to_name.get(mlbam_id)
            if name is None:
                print(f"[WARN] No name found for MLBAM ID {mlbam_id}.")
                name = str(mlbam_id)
            names.append(name)
        return names

    return _resolve(home_ids), _resolve(away_ids)


# ── Statcast: starting pitchers ───────────────────────────────────────────────

def _statcast_pitchers(
    savant_date: str,
    home_abbr: str,
    away_abbr: str,
) -> tuple[str | None, str | None] | None:
    """
    Identify the starting pitcher for each team from statcast data.
    Returns (home_pitcher_name, away_pitcher_name) or None on failure.

    Home pitcher throws in the Top half; away pitcher throws in the Bot half.
    """
    results = {}
    for team, side, inning_half in (
        (home_abbr, "home", "Top"),
        (away_abbr, "away", "Bot"),
    ):
        try:
            raw = pybaseball.statcast(
                start_dt=savant_date,
                end_dt=savant_date,
                team=team.upper(),
                verbose=False,
            )
        except Exception as exc:
            print(f"[WARN] Statcast pitcher fetch failed for {team}: {exc}")
            results[side] = None
            continue

        if raw is None or raw.empty:
            print(f"[WARN] Statcast returned no pitching data for {team} on {savant_date}.")
            results[side] = None
            continue

        raw = raw.copy()
        if "inning_topbot" in raw.columns:
            raw["inning_topbot_norm"] = raw["inning_topbot"].str.strip().str.title()
            inning1 = raw[
                (raw["inning"] == 1) &
                (raw["inning_topbot_norm"] == inning_half)
            ]
            print(f"[DEBUG] Statcast pitcher ({side}): {len(inning1)} inning-1 "
                  f"'{inning_half}' rows for {team}.")
        else:
            inning1 = raw[raw["inning"] == 1]
            print(f"[WARN] inning_topbot missing; using all inning-1 rows for {team}.")

        if inning1.empty:
            print(f"[WARN] No inning-1 rows for {team} pitcher.")
            results[side] = None
            continue

        sort_cols = [c for c in ("at_bat_number", "pitch_number") if c in inning1.columns]
        if sort_cols:
            inning1 = inning1.sort_values(sort_cols)

        mlbam_id = int(inning1.iloc[0]["pitcher"])
        try:
            lookup = pybaseball.playerid_reverse_lookup([mlbam_id], key_type="mlbam")
            if lookup.empty:
                print(f"[WARN] No Chadwick record for pitcher MLBAM {mlbam_id}.")
                results[side] = None
            else:
                row = lookup.iloc[0]
                results[side] = _chadwick_name(row["name_first"], row["name_last"])
                print(f"[DEBUG] Statcast {side} pitcher: {results[side]} (MLBAM {mlbam_id})")
        except Exception as exc:
            print(f"[WARN] playerid_reverse_lookup failed for pitcher {mlbam_id}: {exc}")
            results[side] = None

    home_p = results.get("home")
    away_p = results.get("away")
    if home_p is None and away_p is None:
        return None
    return home_p, away_p


# ── Website fallback: fetch and parse lineup page ────────────────────────────

def _fetch_lineup_soup(savant_date: str) -> BeautifulSoup | None:
    url = _LINEUP_BASE.format(date=savant_date)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as exc:
        print(f"[WARN] Could not fetch lineup page {url}: {exc}")
        return None


def _find_matchup_div(
    soup: BeautifulSoup,
    home_abbr: str,
    away_abbr: str,
):
    """
    Locate the matchup div for the given home/away pair using the
    starting-lineups__teams--away-head / home-head text ("SD Lineup").
    """
    for matchup in soup.find_all("div", class_="starting-lineups__matchup"):
        header = matchup.find("div", class_="starting-lineups__teams--header")
        if not header:
            continue
        away_head = header.find("div", class_="starting-lineups__teams--away-head")
        home_head = header.find("div", class_="starting-lineups__teams--home-head")
        if not away_head or not home_head:
            continue
        page_away = away_head.get_text(strip=True).replace("Lineup", "").strip().upper()
        page_home = home_head.get_text(strip=True).replace("Lineup", "").strip().upper()
        if page_home == home_abbr.upper() and page_away == away_abbr.upper():
            return matchup
    return None


def _website_batters(
    matchup_div,
    home_abbr: str,
    away_abbr: str,
) -> tuple[list[str], list[str]] | None:
    """
    Extract ordered batter names from the lineup page matchup div.
    Uses the visible link text (e.g. 'J.P. Crawford') rather than the slug.
    Returns (home_batters, away_batters) or None.
    """
    away_ol = matchup_div.find("ol", class_="starting-lineups__team--away")
    home_ol = matchup_div.find("ol", class_="starting-lineups__team--home")

    def _names_from_ol(ol) -> list[str]:
        if ol is None:
            return []
        names = []
        for li in ol.find_all("li", class_="starting-lineups__player"):
            a = li.find("a", class_="starting-lineups__player--link")
            if not a:
                continue
            text = a.get_text(strip=True)
            if text:
                names.append(_clean_display_name(text))
            elif a.get("href"):
                # Visible text missing — fall back to slug
                names.append(_slug_to_display_name(a["href"]))
        return names

    home_batters = _names_from_ol(home_ol)
    away_batters = _names_from_ol(away_ol)

    print(f"[DEBUG] Website batters — home ({len(home_batters)}): {home_batters[:2]}... "
          f"away ({len(away_batters)}): {away_batters[:2]}...")

    if not home_batters and not away_batters:
        print(f"[WARN] Website: no batter lists found for {away_abbr} @ {home_abbr}.")
        return None
    return home_batters, away_batters


def _website_pitchers(
    matchup_div,
    home_abbr: str,
    away_abbr: str,
) -> tuple[str | None, str | None]:
    """
    Extract the starting pitcher names from the lineup page matchup div.
    Skips empty placeholder summary divs (there is often a blank first div).
    Returns (home_pitcher, away_pitcher); either may be None if not yet posted.
    """
    overview = matchup_div.find("div", class_="starting-lineups__pitcher-overview")
    if not overview:
        print(f"[WARN] Website: pitcher overview not found for {away_abbr} @ {home_abbr}.")
        return None, None

    all_summaries = overview.find_all("div", class_="starting-lineups__pitcher-summary")
    populated = [s for s in all_summaries
                 if s.find("a", class_="starting-lineups__pitcher--link")]
    print(f"[DEBUG] Website: {len(all_summaries)} pitcher-summary divs, "
          f"{len(populated)} populated.")

    def _name_from_summary(summary) -> str | None:
        link = summary.find("a", class_="starting-lineups__pitcher--link")
        if not link:
            return None
        text = link.get_text(strip=True)
        if text:
            return _clean_display_name(text)
        if link.get("href"):
            return _slug_to_display_name(link["href"])
        return None

    # populated[0] = away pitcher, populated[1] = home pitcher
    away_p = _name_from_summary(populated[0]) if len(populated) > 0 else None
    home_p = _name_from_summary(populated[1]) if len(populated) > 1 else None

    print(f"[DEBUG] Website pitchers — home: {home_p}, away: {away_p}")
    return home_p, away_p


# ── Main public function ──────────────────────────────────────────────────────

def get_game_rosters(
    matchup: tuple[str, str],
    date: str,
) -> tuple[
    tuple[list[str], list[str]],
    tuple[list[str], list[str]],
]:
    """
    Retrieve the starting lineup for both teams in an MLB game.

    Parameters
    ----------
    matchup : (home_abbr, away_abbr)  e.g. ("SD", "LAD")
    date    : MM-DD-YYYY              e.g. "05-18-2026"

    Returns
    -------
    batters  : (home_batter_list, away_batter_list)
               Each list contains 9 full-name strings in batting order.
               May be shorter if lineup is not fully posted.
    pitchers : (home_pitcher_list, away_pitcher_list)
               Each list contains 1 full-name string (the starting pitcher),
               or is empty if not yet available.
    """
    home_abbr, away_abbr = matchup[0].upper().strip(), matchup[1].upper().strip()

    _, savant_date = _parse_date(date)
    print(f"\n{'='*60}")
    print(f"get_game_rosters: {away_abbr} @ {home_abbr} on {savant_date}")
    print(f"{'='*60}")

    home_batters: list[str] = []
    away_batters: list[str] = []
    home_pitcher: str | None = None
    away_pitcher: str | None = None

    # ── Attempt 1: statcast ──────────────────────────────────────────────────
    print("\n[1] Trying statcast for batters...")
    statcast_batters = _statcast_batters(savant_date, home_abbr, away_abbr)
    if statcast_batters is not None:
        home_batters, away_batters = statcast_batters
        print(f"    ✓ Statcast batters: home={len(home_batters)}, away={len(away_batters)}")
    else:
        print("    ✗ Statcast batters unavailable.")

    print("\n[2] Trying statcast for pitchers...")
    statcast_pitchers = _statcast_pitchers(savant_date, home_abbr, away_abbr)
    if statcast_pitchers is not None:
        home_pitcher, away_pitcher = statcast_pitchers
        print(f"    ✓ Statcast pitchers: home={home_pitcher}, away={away_pitcher}")
    else:
        print("    ✗ Statcast pitchers unavailable.")

    # ── Attempt 2: website fallback for anything missing ────────────────────
    needs_website = (
        len(home_batters) < 9 or len(away_batters) < 9 or
        home_pitcher is None or away_pitcher is None
    )

    if needs_website:
        print(f"\n[3] Fetching MLB lineup website for missing data...")
        soup = _fetch_lineup_soup(savant_date)
        if soup is None:
            print("    ✗ Could not reach lineup website.")
        else:
            matchup_div = _find_matchup_div(soup, home_abbr, away_abbr)
            if matchup_div is None:
                print(f"    ✗ Matchup {away_abbr} @ {home_abbr} not found on website.")
            else:
                # Fill missing batters from website
                if len(home_batters) < 9 or len(away_batters) < 9:
                    web_batters = _website_batters(matchup_div, home_abbr, away_abbr)
                    if web_batters is not None:
                        wb_home, wb_away = web_batters
                        if len(home_batters) < 9 and wb_home:
                            home_batters = wb_home
                            print(f"    ✓ Website home batters: {len(home_batters)}")
                        if len(away_batters) < 9 and wb_away:
                            away_batters = wb_away
                            print(f"    ✓ Website away batters: {len(away_batters)}")

                # Fill missing pitchers from website
                if home_pitcher is None or away_pitcher is None:
                    web_home_p, web_away_p = _website_pitchers(
                        matchup_div, home_abbr, away_abbr
                    )
                    if home_pitcher is None and web_home_p:
                        home_pitcher = web_home_p
                        print(f"    ✓ Website home pitcher: {home_pitcher}")
                    if away_pitcher is None and web_away_p:
                        away_pitcher = web_away_p
                        print(f"    ✓ Website away pitcher: {away_pitcher}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Result: {away_abbr} @ {home_abbr}")
    print(f"  Home batters  ({len(home_batters)}): {home_batters}")
    print(f"  Away batters  ({len(away_batters)}): {away_batters}")
    print(f"  Home pitcher: {home_pitcher}")
    print(f"  Away pitcher: {away_pitcher}")
    print(f"{'─'*60}\n")

    home_pitcher_list = [home_pitcher] if home_pitcher else []
    away_pitcher_list = [away_pitcher] if away_pitcher else []

    return (home_batters, away_batters), (home_pitcher_list, away_pitcher_list)


def get_player_stats(csv_path: str, target_name: str) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    
    # 1. Normalize the target name from the input
    # Decompose the accents, encode to ascii (ignoring the accents), decode back to string, strip, and lowercase
    clean_target = unicodedata.normalize('NFKD', target_name).encode('ascii', 'ignore').decode('utf-8').strip().lower()
    
    # 2. Normalize the 'Name' column in the dataframe using pandas' vectorized string methods
    clean_names = (
        df['Name']
        .str.normalize('NFKD')
        .str.encode('ascii', errors='ignore')
        .str.decode('utf-8')
        .str.strip()
        .str.lower()
    )
    
    # 3. Match the cleaned column against the cleaned target
    person_row = df[clean_names == clean_target]
    
    if person_row.empty:
        return {}
        
    # 4. Extract the stats
    stat_cols = [col for col in df.columns if col.isupper() and 2 <= len(col) <= 3]
    
    return person_row[stat_cols].iloc[0].astype(float).to_dict()
# ── Quick demo when run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    matchup = (sys.argv[1], sys.argv[2]) if len(sys.argv) >= 3 else ("SD", "LAD")
    date    = sys.argv[3] if len(sys.argv) >= 4 else "05-18-2026"
    batters, pitchers = get_game_rosters(matchup, date)
    home_b, away_b = batters
    home_p, away_p = pitchers
    print(f"{matchup[1]} (away) lineup: {away_b}")
    print(f"{matchup[0]} (home) lineup: {home_b}")
    print(f"Pitchers: home={home_p}, away={away_p}")
