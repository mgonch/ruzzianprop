"""
File-based data collector.

Reads pre-collected tweet / user data from JSON or CSV files,
normalising it into the same schema used by TwitterCollector.

Expected JSON format (list of objects):
  [
    {
      "id": "...",
      "text": "...",
      "author": {
        "id": "...", "username": "...", "name": "...",
        "created_at": "2020-01-01T00:00:00Z",
        "followers_count": 123, "following_count": 45,
        "tweet_count": 678, "description": "...", ...
      },
      "created_at": "...",
      "retweet_count": 0, "like_count": 0, ...
    },
    ...
  ]
"""

import json
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class FileCollector:
    """Load tweet data from a local JSON or CSV file."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def load(self, path: str) -> list[dict]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        suffix = p.suffix.lower()
        if suffix == ".json":
            return self._load_json(p)
        elif suffix in (".csv",):
            return self._load_csv(p)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .json or .csv")

    # ------------------------------------------------------------------

    def _load_json(self, path: Path) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raw = [raw]

        records = []
        for item in raw:
            records.append(self._normalise(item))
        logger.info("Loaded %d records from %s", len(records), path)
        return records

    def _load_csv(self, path: Path) -> list[dict]:
        records = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(self._normalise_flat(row))
        logger.info("Loaded %d records from %s", len(records), path)
        return records

    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(item: dict) -> dict:
        """Normalise a nested JSON record."""
        author = item.get("author") or {}
        created_at = author.get("created_at")
        age_days = None
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - dt).days
            except ValueError:
                pass

        return {
            "id": str(item.get("id", "")),
            "text": item.get("text", ""),
            "author_id": str(author.get("id", item.get("author_id", ""))),
            "created_at": item.get("created_at"),
            "lang": item.get("lang"),
            "retweet_count": int(item.get("retweet_count", 0) or 0),
            "reply_count": int(item.get("reply_count", 0) or 0),
            "like_count": int(item.get("like_count", 0) or 0),
            "quote_count": int(item.get("quote_count", 0) or 0),
            "referenced_tweets": item.get("referenced_tweets", []),
            "entities": item.get("entities", {}),
            "source": item.get("source"),
            "author": {
                "id": str(author.get("id", "")),
                "name": author.get("name", ""),
                "username": author.get("username", ""),
                "created_at": created_at,
                "account_age_days": author.get("account_age_days", age_days),
                "description": author.get("description", ""),
                "location": author.get("location", ""),
                "profile_image_url": author.get("profile_image_url"),
                "verified": bool(author.get("verified", False)),
                "protected": bool(author.get("protected", False)),
                "followers_count": int(author.get("followers_count", 0) or 0),
                "following_count": int(author.get("following_count", 0) or 0),
                "tweet_count": int(author.get("tweet_count", 0) or 0),
                "listed_count": int(author.get("listed_count", 0) or 0),
            },
        }

    @staticmethod
    def _normalise_flat(row: dict) -> dict:
        """Normalise a flat CSV row where author fields are prefixed author_*."""
        created_at = row.get("author_created_at") or row.get("created_at_account")
        age_days = None
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - dt).days
            except ValueError:
                pass

        def intval(key: str) -> int:
            return int(row.get(key, 0) or 0)

        return {
            "id": str(row.get("id", row.get("tweet_id", ""))),
            "text": row.get("text", row.get("full_text", "")),
            "author_id": str(row.get("author_id", row.get("user_id", ""))),
            "created_at": row.get("created_at"),
            "lang": row.get("lang"),
            "retweet_count": intval("retweet_count"),
            "reply_count": intval("reply_count"),
            "like_count": intval("like_count"),
            "quote_count": intval("quote_count"),
            "referenced_tweets": [],
            "entities": {},
            "source": row.get("source"),
            "author": {
                "id": str(row.get("author_id", row.get("user_id", ""))),
                "name": row.get("author_name", row.get("name", "")),
                "username": row.get("author_username", row.get("username", row.get("screen_name", ""))),
                "created_at": created_at,
                "account_age_days": age_days,
                "description": row.get("author_description", row.get("description", "")),
                "location": row.get("author_location", row.get("location", "")),
                "profile_image_url": row.get("author_profile_image_url"),
                "verified": row.get("author_verified", "false").lower() == "true",
                "protected": row.get("author_protected", "false").lower() == "true",
                "followers_count": intval("author_followers_count"),
                "following_count": intval("author_following_count"),
                "tweet_count": intval("author_tweet_count"),
                "listed_count": intval("author_listed_count"),
            },
        }
