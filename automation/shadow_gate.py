#!/usr/bin/env python3
"""Dependency-free campaign and hourly de-duplication gate for cloud runners."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "shadow_gth10.json"


def _load() -> tuple[dict, Path]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    campaign_root = ROOT / "data" / "shadow" / config["campaign_id"]
    return config, campaign_root


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("campaign timestamp is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def _full_issue_in_hour(campaign_root: Path, now: datetime, expected: int) -> bool:
    folder = campaign_root / "runs" / now.strftime("%Y-%m-%d")
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            invoked = _parse(record["automation"]["runner_invoked_at_utc"])
            same_hour = invoked.replace(minute=0, second=0, microsecond=0) == now.replace(
                minute=0, second=0, microsecond=0
            )
            if same_hour and len(record.get("forecasts") or []) >= expected:
                return True
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return False


def gate(mode: str, now: datetime) -> tuple[bool, str]:
    config, campaign_root = _load()
    state = campaign_root / "campaign.json"
    if not state.exists():
        return (mode == "forecast", "awaiting_first_successful_forecast")
    campaign = json.loads(state.read_text(encoding="utf-8"))
    end = _parse(campaign["ends_at_utc"])
    cutoff = end if mode == "forecast" else end + timedelta(days=config["scoring_tail_days"])
    if now >= cutoff:
        return False, f"window_ended_at_{cutoff.isoformat()}"
    if mode == "forecast":
        expected = len(config["station_keys"]) * 24
        if _full_issue_in_hour(campaign_root, now, expected):
            return False, "complete_issue_already_exists_in_this_utc_hour"
    return True, f"window_open_until_{cutoff.isoformat()}"


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "forecast"
    if mode not in {"forecast", "score"}:
        raise SystemExit("mode must be forecast or score")
    enabled, reason = gate(mode, datetime.now(timezone.utc))
    print(f"SHADOW_GATE mode={mode} enabled={enabled} reason={reason}")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"enabled={str(enabled).lower()}\n")


if __name__ == "__main__":
    main()
