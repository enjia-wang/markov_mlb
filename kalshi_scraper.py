"""
build_mlb_lines_db.py  —  kalshi_python_sync 3.2.0

Reads a CSV (columns: Home, Away, Date, optionally GameTime) and records
pre-game Kalshi implied probabilities for four market types per game:

  a) Moneyline            series: KXMLBGAME
  b) Over 8.5 total runs  series: KXMLBTOTAL    (yes_sub_title contains "8.5")
  c) Over 4.5 F5 runs     series: KXMLBF5TOTAL  (yes_sub_title contains "4.5")
  d) Run in 1st inning    series: KXMLBRFI

kalshi.env must contain:
  KALSHI_API_KEY_ID=<your key id>
  KALSHI_PRIVATE_KEY_PATH=<path to your .key PEM file>

Verified against kalshi_python_sync 3.2.0 source:
  - KalshiClient wraps all APIs and forwards method calls via __getattr__
  - Configuration takes api_key_id / private_key_pem as direct attributes
  - get_events(series_ticker, status, limit, cursor) → GetEventsResponse
      .events  list[EventData]   .cursor  str
  - get_markets(event_ticker, status='settled', limit, cursor) → GetMarketsResponse
      .markets  list[Market]     .cursor  str
      Market.yes_sub_title  str
      Market.ticker         str
  - get_market_candlesticks(series_ticker, ticker, start_ts, end_ts, period_interval)
      period_interval: 1=1min, 60=1hr, 1440=1day
      → GetMarketCandlesticksResponse  .candlesticks  list[MarketCandlestick]
      MarketCandlestick.yes_bid  BidAskDistribution  .close  int (CENTS)
      MarketCandlestick.price    PriceDistribution   .close  int|None (CENTS)
  - Implied probability = yes_bid.close / 100  (e.g. 62 cents → 0.62)
"""

import os
import time
import traceback
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import kalshi_python_sync as k

# ── Auth ───────────────────────────────────────────────────────────────────────
load_dotenv("kalshi.env")

_key_id   = os.environ["KALSHI_API_KEY_ID"]
_key_path = os.environ["KALSHI_PRIVATE_KEY_PATH"]
with open(_key_path, "r") as f:
    _key_pem = f.read()

conf = k.Configuration(host="https://api.elections.kalshi.com/trade-api/v2")
conf.api_key_id      = _key_id
conf.private_key_pem = _key_pem

client = k.KalshiClient(configuration=conf)

# Baseball season = EDT (UTC-4)
ET = timezone(timedelta(hours=-4))

# ── Series tickers ─────────────────────────────────────────────────────────────
SERIES = {
    "ml":  "KXMLBGAME",
    "o85": "KXMLBTOTAL",
    "f5":  "KXMLBF5TOTAL",
    "rfi": "KXMLBRFI",
}

# Substring to match in yes_sub_title for multi-strike markets; None = take first
SUBTAG = {
    "ml":  None,
    "o85": "8.5",
    "f5":  "4.5",
    "rfi": None,
}

TEAM_ABBREV = {
    "Arizona Diamondbacks":  "ARI",  "Atlanta Braves":       "ATL",
    "Baltimore Orioles":     "BAL",  "Boston Red Sox":       "BOS",
    "Chicago Cubs":          "CHC",  "Chicago White Sox":    "CWS",
    "Cincinnati Reds":       "CIN",  "Cleveland Guardians":  "CLE",
    "Colorado Rockies":      "COL",  "Detroit Tigers":       "DET",
    "Houston Astros":        "HOU",  "Kansas City Royals":   "KC",
    "Los Angeles Angels":    "LAA",  "Los Angeles Dodgers":  "LAD",
    "Miami Marlins":         "MIA",  "Milwaukee Brewers":    "MIL",
    "Minnesota Twins":       "MIN",  "New York Mets":        "NYM",
    "New York Yankees":      "NYY",  "Oakland Athletics":    "ATH",
    "Philadelphia Phillies": "PHI",  "Pittsburgh Pirates":   "PIT",
    "San Diego Padres":      "SD",   "San Francisco Giants": "SF",
    "Seattle Mariners":      "SEA",  "St. Louis Cardinals":  "STL",
    "Tampa Bay Rays":        "TB",   "Texas Rangers":        "TEX",
    "Toronto Blue Jays":     "TOR",  "Washington Nationals": "WSH",
}

def to_abbrev(name: str) -> str:
    n = name.strip()
    if n.upper() in TEAM_ABBREV.values():
        return n.upper()
    result = TEAM_ABBREV.get(n)
    if not result:
        raise ValueError(
            f"Unknown team: {n!r}. Add it to TEAM_ABBREV.\n"
            f"Known names: {sorted(TEAM_ABBREV.keys())}"
        )
    return result


# ── Step 1: find the event ticker ──────────────────────────────────────────────
# get_events(series_ticker, status, limit, cursor) → GetEventsResponse
#   .events  list[EventData]  each has .event_ticker (str)
#   .cursor  str (empty string when no more pages)
#
# Event ticker format: {SERIES}-{YYMMDD}[HHMM]{AWAY}{HOME}  e.g. KXMLBGAME-260425190NYYBOS
# We match date+away+home as a case-insensitive substring.

def find_event_ticker(series: str, game_dt: datetime, away_abbr: str, home_abbr: str) -> str | None:
    date_str = game_dt.strftime("%y%m%d")           # e.g. "260425"
    fragment = f"{date_str}{away_abbr}{home_abbr}"  # e.g. "260425NYYBOS"

    print(f"      [find_event] series={series}  looking for fragment '{fragment}'")

    cursor = None
    page   = 0
    while True:
        page += 1
        try:
            resp = client.get_events(
                series_ticker=series,
                status="settled",
                limit=200,
                cursor=cursor,
            )
        except k.ApiException as e:
            print(f"      ERROR get_events(series={series}): HTTP {e.status}  {e.reason}")
            print(f"      Body: {e.body}")
            return None
        except Exception as e:
            print(f"      ERROR get_events(series={series}): {e}")
            traceback.print_exc()
            return None

        events = resp.events or []
        print(f"      Page {page}: {len(events)} events returned")

        for ev in events:
            if fragment.upper() in ev.event_ticker.upper():
                print(f"      MATCH: {ev.event_ticker}")
                return ev.event_ticker

        # Paginate — cursor is empty string (not None) when exhausted
        cursor = resp.cursor
        if not cursor or not events:
            break
        time.sleep(0.1)

    # Debug: show last few tickers so mismatches are visible
    if events:
        sample = [ev.event_ticker for ev in events[-5:]]
        print(f"      NOT FOUND. Last 5 tickers seen in {series}: {sample}")
        print(f"      HINT: check team abbreviation and date format match above tickers.")
    else:
        print(f"      NOT FOUND. No events returned at all for series={series}")
        print(f"      HINT: series ticker may be wrong, or no settled events exist yet.")
    return None


# ── Step 2: get markets for that event ────────────────────────────────────────
# get_markets(event_ticker, status='settled', limit, cursor) → GetMarketsResponse
#   .markets  list[Market]   Market.ticker, Market.yes_sub_title, Market.event_ticker

def get_markets_for_event(event_ticker: str) -> list:
    print(f"      [get_markets] event_ticker={event_ticker}")
    all_markets = []
    cursor = None
    while True:
        try:
            resp = client.get_markets(
                event_ticker=event_ticker,
                status="settled",
                limit=100,
                cursor=cursor,
                mve_filter="exclude",
            )
        except k.ApiException as e:
            print(f"      ERROR get_markets(event={event_ticker}): HTTP {e.status}  {e.reason}")
            print(f"      Body: {e.body}")
            return []
        except Exception as e:
            print(f"      ERROR get_markets(event={event_ticker}): {e}")
            traceback.print_exc()
            return []

        markets = resp.markets or []
        all_markets.extend(markets)

        cursor = resp.cursor
        if not cursor or not markets:
            break
        time.sleep(0.05)

    print(f"      Found {len(all_markets)} market(s):")
    for m in all_markets:
        print(f"        ticker={m.ticker}  yes_sub_title={m.yes_sub_title!r}")
    return all_markets


# ── Step 3: pick the right sub-market ─────────────────────────────────────────

def pick_market(markets: list, subtag: str | None, event_ticker: str):
    if not markets:
        print(f"      [pick_market] No markets for {event_ticker}")
        return None

    if subtag is None:
        m = markets[0]
        print(f"      [pick_market] Using first: {m.ticker}")
        return m

    for m in markets:
        yst = m.yes_sub_title or ""
        rp  = m.rules_primary  or ""
        if subtag in yst or subtag in rp:
            print(f"      [pick_market] Matched '{subtag}' → {m.ticker}")
            return m

    print(
        f"      [pick_market] WARNING: '{subtag}' not found.\n"
        f"      Available yes_sub_titles: {[m.yes_sub_title for m in markets]}"
    )
    return None


# ── Step 4: get pre-game price from candlesticks ───────────────────────────────
# get_market_candlesticks(series_ticker, ticker, start_ts, end_ts, period_interval)
#   period_interval: 1 (1-min), 60 (1-hr), 1440 (1-day)
#   → GetMarketCandlesticksResponse  .candlesticks  list[MarketCandlestick]
#   MarketCandlestick.yes_bid.close  →  int in CENTS  →  divide by 100 for prob
#   MarketCandlestick.price.close    →  int|None in CENTS (None if no trades that period)

def get_pregame_prob(series_ticker: str, market_ticker: str, game_ts: int) -> float | None:
    for lookback_minutes in (60, 180, 720):
        start_ts = game_ts - (lookback_minutes * 60)
        print(f"      [candlesticks] {market_ticker}  lookback={lookback_minutes}min  "
              f"[{start_ts} → {game_ts}]")

        try:
            resp = client.get_market_candlesticks(
                series_ticker=series_ticker,
                ticker=market_ticker,
                start_ts=start_ts,
                end_ts=game_ts,
                period_interval=1,
            )
        except k.ApiException as e:
            print(f"      ERROR get_market_candlesticks({market_ticker}): HTTP {e.status}  {e.reason}")
            print(f"      Body: {e.body}")
            return None
        except Exception as e:
            print(f"      ERROR get_market_candlesticks({market_ticker}): {e}")
            traceback.print_exc()
            return None

        candles = resp.candlesticks or []
        print(f"      Got {len(candles)} candle(s)")

        if candles:
            last = candles[-1]
            yb   = last.yes_bid
            pr   = last.price
            print(f"      Last candle: end_ts={last.end_period_ts}  "
                  f"yes_bid.close={yb.close if yb else 'N/A'}¢  "
                  f"price.close={pr.close if pr else 'N/A'}¢")

            # yes_bid.close is always present (BidAskDistribution is non-optional)
            # price.close may be None if no trades occurred that period
            if yb and yb.close is not None:
                return yb.close / 100.0     # cents → probability

            if pr and pr.close is not None:
                return pr.close / 100.0

        time.sleep(0.1)

    print(f"      No candle data in any lookback window for {market_ticker}")
    return None


# ── Per-game orchestration ─────────────────────────────────────────────────────

def collect_game_lines(home: str, away: str, game_dt: datetime) -> dict:
    game_ts   = int(game_dt.timestamp())
    away_abbr = to_abbrev(away)
    home_abbr = to_abbrev(home)

    row = {"ml_prob": None, "o85_prob": None, "f5_o45_prob": None, "rfi_prob": None}

    configs = [
        ("ml",  "ml_prob"),
        ("o85", "o85_prob"),
        ("f5",  "f5_o45_prob"),
        ("rfi", "rfi_prob"),
    ]

    for key, col in configs:
        series = SERIES[key]
        subtag = SUBTAG[key]

        print(f"\n    ── {series}  subtag={subtag!r} ──")

        event_ticker = find_event_ticker(series, game_dt, away_abbr, home_abbr)
        if not event_ticker:
            continue

        markets = get_markets_for_event(event_ticker)
        if not markets:
            continue

        mkt = pick_market(markets, subtag, event_ticker)
        if not mkt:
            continue

        prob = get_pregame_prob(series, mkt.ticker, game_ts)
        if prob is not None:
            print(f"      → {prob:.4f}  ({prob*100:.1f}%)")
            row[col] = prob
        else:
            print(f"      → no price data found")

        time.sleep(0.15)

    return row


# ── Main ───────────────────────────────────────────────────────────────────────

def build_mlb_lines_db(
    csv_path: str,
    date_range: tuple,
    output_path: str = "mlb_kalshi_lines.csv",
    time_col: str | None = "GameTime",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    csv_path    : CSV with Home, Away, Date columns. Optional GameTime (HH:MM ET).
    date_range  : ("YYYY-MM-DD", "YYYY-MM-DD") inclusive.
    output_path : Where to write the enriched CSV.
    time_col    : Column holding game start time (HH:MM). None → default 19:05 ET.

    Notes
    -----
    - ml_prob is P(AWAY wins) — that's how Kalshi's YES contract is defined.
    - All probabilities are 0.0–1.0.
    - Checkpoint saves every 5 games so a crash doesn't lose progress.
    """
    start_date = datetime.strptime(date_range[0], "%Y-%m-%d").date()
    end_date   = datetime.strptime(date_range[1], "%Y-%m-%d").date()

    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    subset = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy().reset_index(drop=True)

    if subset.empty:
        print(f"No games found between {start_date} and {end_date}.")
        return subset

    print(f"\n{'='*60}")
    print(f"Processing {len(subset)} games  [{start_date} → {end_date}]")
    print(f"{'='*60}")

    all_rows = []

    for i, game in subset.iterrows():
        date = game["Date"]
        home = str(game["Home"]).strip()
        away = str(game["Away"]).strip()

        # Parse game time (default 19:05 ET if missing)
        h, m = 19, 5
        if time_col and time_col in game and pd.notna(game.get(time_col)):
            raw = str(game[time_col]).strip()
            for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p"):
                try:
                    t  = datetime.strptime(raw, fmt)
                    h, m = t.hour, t.minute
                    break
                except ValueError:
                    continue
            else:
                print(f"  WARNING: could not parse GameTime '{raw}', using 19:05")

        game_dt = datetime(date.year, date.month, date.day, h, m, tzinfo=ET)

        print(f"\n{'─'*60}")
        print(f"Game {i+1}/{len(subset)}: {away} @ {home}  "
              f"{game_dt.strftime('%Y-%m-%d %H:%M ET')}  (unix={int(game_dt.timestamp())})")

        try:
            row = collect_game_lines(home, away, game_dt)
        except Exception as e:
            print(f"  FATAL ERROR: {e}")
            traceback.print_exc()
            row = {"ml_prob": None, "o85_prob": None, "f5_o45_prob": None, "rfi_prob": None}

        print(f"\n  RESULT → ml={row['ml_prob']}  o85={row['o85_prob']}  "
              f"f5={row['f5_o45_prob']}  rfi={row['rfi_prob']}")
        all_rows.append(row)

        # Checkpoint save every 5 games
        if (i + 1) % 5 == 0:
            partial = subset.iloc[:i+1].copy()
            partial[["ml_prob","o85_prob","f5_o45_prob","rfi_prob"]] = pd.DataFrame(all_rows)
            partial.to_csv(output_path, index=False)
            print(f"\n  [checkpoint] {i+1} games saved → {output_path}")

    lines_df = pd.DataFrame(all_rows)
    result   = subset.copy()
    result[["ml_prob", "o85_prob", "f5_o45_prob", "rfi_prob"]] = lines_df
    result.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"Complete. {len(result)} rows → {output_path}")
    filled = result[["ml_prob","o85_prob","f5_o45_prob","rfi_prob"]].notna().sum()
    print(f"Fill rates: {filled.to_dict()}")
    return result

if __name__ == "__main__":
    csv_with_games = "mlb_archive_zips_2.csv"
    df = build_mlb_lines_db(
        csv_path   = csv_with_games,
        date_range = ("2026-05-01", "2026-05-02"),
        output_path= csv_with_games,
        time_col   = "GameTime",   # or None if you don't have this column
    )
    print(df[["Date","Away","Home","ml_prob","o85_prob","f5_o45_prob","rfi_prob"]].head(10))
