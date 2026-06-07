"""
Scrape MLB moneyline implied probabilities from Kalshi using pykalshi.

Kalshi's MLB game-winner markets use the format:
  KXMLB-YYMMDD-<AWAYTEAM>-<HOMETEAM>
e.g., KXMLB-260528-NYY-BOS  (Yankees @ Red Sox on May 28, 2026)

The YES price in cents IS the implied probability (e.g., 62¢ = 62%).
"""

from pykalshi import KalshiClient, MarketStatus
from datetime import date
from dotenv import load_dotenv

load_dotenv("kalshi.env") 

client = KalshiClient.from_env()

# ── Step 1: Find all open MLB markets for today ───────────────────────────────

# today = date.today().strftime("%y%m%d")   # e.g., "260528"
today = "260529"

mlb_markets = client.get_markets(
    status=MarketStatus.OPEN,
    series_ticker="KXMLB",    # MLB game-winner series
    limit=50,
)

# Filter to today's games only (ticker contains today's date)
todays_games = [m for m in mlb_markets if today in m.ticker]

print(f"Found {len(todays_games)} MLB markets for {date.today()}\n")

# ── Step 2: Extract implied moneyline probabilities ───────────────────────────

results = []

for market in todays_games:
    # yes_bid and yes_ask are in dollars (e.g., 0.62)
    # The midpoint is the market's best estimate of win probability
    yes_bid  = market.yes_bid_dollars   # buyer's best price
    yes_ask  = market.yes_ask_dollars   # seller's best price

    if yes_bid is not None and yes_ask is not None:
        mid = (yes_bid + yes_ask) / 2
    elif yes_bid is not None:
        mid = yes_bid
    elif yes_ask is not None:
        mid = yes_ask
    else:
        mid = None

    results.append({
        "ticker":       market.ticker,
        "title":        market.title,
        "yes_bid":      yes_bid,
        "yes_ask":      yes_ask,
        "implied_prob": mid,
        "volume":       market.volume_fp,
    })

# ── Step 3: Print a clean summary ────────────────────────────────────────────

print(f"{'Matchup':<45} {'Win Prob':>9}  {'Bid':>6}  {'Ask':>6}  {'Volume':>8}")
print("-" * 80)

for r in sorted(results, key=lambda x: x["implied_prob"] or 0, reverse=True):
    prob_str   = f"{r['implied_prob']*100:.1f}%" if r["implied_prob"] else "N/A"
    bid_str    = f"${r['yes_bid']:.2f}"           if r["yes_bid"]     else "N/A"
    ask_str    = f"${r['yes_ask']:.2f}"           if r["yes_ask"]     else "N/A"
    volume_str = f"{r['volume']:.0f}"             if r["volume"]      else "N/A"
    print(f"{r['title']:<45} {prob_str:>9}  {bid_str:>6}  {ask_str:>6}  {volume_str:>8}")