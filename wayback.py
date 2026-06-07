"""
fetch_wayback_tweets.py
-----------------------
Fetch archived tweets for a Twitter/X user from the Wayback Machine
and export them as a styled HTML file.

Dependencies:
    pip install waybacktweets

Usage example:
    tweets_to_html("jack", ("01/01/2020", "12/31/2022"))
"""

from __future__ import annotations

from datetime import datetime

from waybacktweets import WaybackTweets, TweetsParser, TweetsExporter
from waybacktweets.api.visualize import HTMLTweetsVisualizer


# Fields to request and display in the exported HTML.
# Remove or add entries to taste; see the full list at:
# https://waybacktweets.claromes.com/field_options.html
FIELD_OPTIONS: list[str] = [
    "archived_timestamp",
    "parsed_archived_timestamp",
    "archived_tweet_url",
    "parsed_archived_tweet_url",
    "original_tweet_url",
    "parsed_tweet_url",
    "available_tweet_text",
    "available_tweet_is_RT",
    "available_tweet_info",
    "archived_mimetype",
    "archived_statuscode",
]


def _mm_dd_yyyy_to_yyyymmdd(date_str: str) -> str:
    """Convert a date string from MM/DD/YYYY to YYYYmmdd (Wayback Machine format).

    Args:
        date_str: Date in MM/DD/YYYY format, e.g. ``"03/05/2020"``.

    Returns:
        Date string in YYYYmmdd format, e.g. ``"20200305"``.

    Raises:
        ValueError: If *date_str* does not match MM/DD/YYYY.
    """
    try:
        dt = datetime.strptime(date_str.strip(), "%m/%d/%Y")
    except ValueError as exc:
        raise ValueError(
            f"Invalid date '{date_str}'. Expected MM/DD/YYYY format, "
            "e.g. '03/05/2020'."
        ) from exc
    return dt.strftime("%Y%m%d")


def tweets_to_html(
    username: str,
    date_range: tuple[str, str],
    output_dir: str = ".",
    field_options: list[str] | None = None,
) -> str:
    """Fetch archived tweets and export them as an HTML file.

    Uses the Wayback Machine CDX API via the ``waybacktweets`` library to
    retrieve snapshots of a user's tweets within the specified date range,
    then writes a self-contained, browser-viewable HTML file.

    Args:
        username:
            Twitter/X handle *without* the leading ``@``, e.g. ``"jack"``.
        date_range:
            A two-element tuple of date strings in **MM/DD/YYYY** format:
            ``(start_date, end_date)``.  Both ends are inclusive.
            Example: ``("01/01/2020", "12/31/2022")``.
        output_dir:
            Directory where the HTML file will be written.  Defaults to the
            current working directory (``"."``).
        field_options:
            List of CDX fields to include in the export.  When omitted the
            module-level ``FIELD_OPTIONS`` list is used.  See the full field
            reference at https://waybacktweets.claromes.com/field_options.html

    Returns:
        Absolute path to the generated HTML file.

    Raises:
        ValueError: If either date string is not in MM/DD/YYYY format.
        RuntimeError: If the Wayback Machine returns no results for the query.

    Example::

        path = tweets_to_html("jack", ("01/01/2020", "12/31/2022"))
        print(f"Saved to: {path}")
    """
    import os

    start_str, end_str = date_range
    timestamp_from = _mm_dd_yyyy_to_yyyymmdd(start_str)
    timestamp_to = _mm_dd_yyyy_to_yyyymmdd(end_str)
    fields = field_options if field_options is not None else FIELD_OPTIONS

    # --- 1. Fetch archived tweet records from the Wayback CDX API ----------
    api = WaybackTweets(
        username=username,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
    )
    archived_tweets = api.get()

    if not archived_tweets:
        raise RuntimeError(
            f"No archived tweets found for @{username} between "
            f"{start_str} and {end_str}."
        )

    # --- 2. Parse the CDX response -----------------------------------------
    parser = TweetsParser(
        archived_tweets_response=archived_tweets,
        username=username,
        field_options=fields,
    )
    parsed_tweets = parser.parse()

    # --- 3. Build the output file path -------------------------------------
    safe_username = username.lstrip("@").lower()
    html_filename = f"{safe_username}_wayback_tweets.html"
    html_file_path = os.path.join(os.path.abspath(output_dir), html_filename)

    # --- 4. Export to HTML via the visualizer ------------------------------
    # TweetsExporter.generate_json() returns the data as a JSON string,
    # which HTMLTweetsVisualizer accepts directly as json_path.
    exporter = TweetsExporter(
        data=parsed_tweets,
        username=username,
        field_options=fields,
    )
    json_data = exporter.generate_json()

    visualizer = HTMLTweetsVisualizer(
        username=username,
        json_path=json_data,          # accepts a JSON string or a file path
        html_file_path=html_file_path,
    )
    html_content = visualizer.generate()
    visualizer.save(html_content)

    print(f"✓ Exported {len(parsed_tweets.get('archived_tweet_url', []))} "
          f"tweet snapshots to:\n  {html_file_path}")

    return html_file_path


# ---------------------------------------------------------------------------
# Quick smoke-test – run this file directly to try it out:
#   python fetch_wayback_tweets.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4:
        user = sys.argv[1]
        start = sys.argv[2]   # MM/DD/YYYY
        end = sys.argv[3]     # MM/DD/YYYY
    else:
        # Sensible demo defaults (a short window to keep the query fast)
        user = "jack"
        start = "01/01/2020"
        end = "03/31/2020"

    output = tweets_to_html(user, (start, end))
    print(f"\nOpen in your browser:\n  file://{output}")