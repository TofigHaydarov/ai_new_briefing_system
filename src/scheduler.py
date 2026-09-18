"""Run the existing daily briefing command on a schedule."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta

from src.cli import generate_briefing


DEFAULT_RUN_TIME = "08:00"


def _seconds_until(run_time: str) -> float:
    hour, minute = (int(part) for part in run_time.split(":", 1))
    now = datetime.now()
    next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if next_run <= now:
        next_run += timedelta(days=1)
    return (next_run - now).total_seconds()


def run_daily() -> None:
    """Generate briefings for all registered users once."""
    generate_briefing(username=None, run_all=True)


def run_scheduler(run_time: str = DEFAULT_RUN_TIME) -> None:
    """Run the briefing every day at the local time given as HH:MM."""
    while True:
        time.sleep(_seconds_until(run_time))
        run_daily()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily AI briefings")
    parser.add_argument(
        "--time",
        default=DEFAULT_RUN_TIME,
        help="Local daily run time in HH:MM format (default: 08:00)",
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Generate all briefings immediately and exit",
    )
    args = parser.parse_args()
    datetime.strptime(args.time, "%H:%M")

    if args.run_now:
        run_daily()
    else:
        run_scheduler(args.time)


if __name__ == "__main__":
    main()