"""
Twitter/X data collector using the Twitter API v2 via Tweepy.

Requires a Twitter Developer account with at least Basic tier access.
Set credentials via environment variables:
  TWITTER_BEARER_TOKEN
  TWITTER_API_KEY
  TWITTER_API_SECRET
  TWITTER_ACCESS_TOKEN
  TWITTER_ACCESS_SECRET
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional

import tweepy

logger = logging.getLogger(__name__)


class TwitterCollector:
    """Collect account and tweet data from the Twitter/X API v2."""

    # Fields requested for each user object
    USER_FIELDS = [
        "id", "name", "username", "created_at", "description",
        "entities", "location", "pinned_tweet_id", "profile_image_url",
        "protected", "public_metrics", "url", "verified",
        "withheld",
    ]

    # Fields requested for each tweet object
    TWEET_FIELDS = [
        "id", "text", "author_id", "created_at", "entities",
        "geo", "in_reply_to_user_id", "lang", "public_metrics",
        "referenced_tweets", "source",
    ]

    EXPANSIONS = [
        "author_id", "referenced_tweets.id", "referenced_tweets.id.author_id",
        "entities.mentions.username",
    ]

    def __init__(self, config: dict):
        self.config = config
        self.client = self._build_client()

    def _build_client(self) -> tweepy.Client:
        bearer = (
            os.environ.get("TWITTER_BEARER_TOKEN")
            or self.config.get("twitter", {}).get("bearer_token", "")
        )
        api_key = (
            os.environ.get("TWITTER_API_KEY")
            or self.config.get("twitter", {}).get("api_key", "")
        )
        api_secret = (
            os.environ.get("TWITTER_API_SECRET")
            or self.config.get("twitter", {}).get("api_secret", "")
        )
        access_token = (
            os.environ.get("TWITTER_ACCESS_TOKEN")
            or self.config.get("twitter", {}).get("access_token", "")
        )
        access_secret = (
            os.environ.get("TWITTER_ACCESS_SECRET")
            or self.config.get("twitter", {}).get("access_secret", "")
        )

        if not bearer and not (api_key and api_secret):
            raise EnvironmentError(
                "Twitter credentials not configured. Set TWITTER_BEARER_TOKEN "
                "or TWITTER_API_KEY/TWITTER_API_SECRET environment variables."
            )

        return tweepy.Client(
            bearer_token=bearer or None,
            consumer_key=api_key or None,
            consumer_secret=api_secret or None,
            access_token=access_token or None,
            access_token_secret=access_secret or None,
            wait_on_rate_limit=True,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search_recent(self, query: str, max_results: int = 100) -> list[dict]:
        """
        Search recent tweets matching *query* and return normalised records.

        Each record contains both tweet-level and author-level data.
        """
        logger.info("Searching recent tweets for query: %s", query)
        results = []
        paginator = tweepy.Paginator(
            self.client.search_recent_tweets,
            query=query,
            tweet_fields=self.TWEET_FIELDS,
            user_fields=self.USER_FIELDS,
            expansions=self.EXPANSIONS,
            max_results=min(max_results, 100),
        ).flatten(limit=max_results)

        users_by_id: dict[str, dict] = {}

        for tweet in paginator:
            record = self._normalise_tweet(tweet)
            results.append(record)

        # Hydrate authors in a second pass
        author_ids = list({r["author_id"] for r in results if r.get("author_id")})
        if author_ids:
            users_by_id = self._fetch_users_by_ids(author_ids)

        for record in results:
            aid = record.get("author_id")
            if aid and aid in users_by_id:
                record["author"] = users_by_id[aid]

        logger.info("Collected %d tweets", len(results))
        return results

    def get_user_timeline(self, username: str, max_results: int = 200) -> list[dict]:
        """Fetch the tweet timeline for a single *username*."""
        logger.info("Fetching timeline for @%s", username)
        user_resp = self.client.get_user(
            username=username,
            user_fields=self.USER_FIELDS,
        )
        if not user_resp.data:
            logger.warning("User @%s not found", username)
            return []

        user = self._normalise_user(user_resp.data)
        tweets = []
        paginator = tweepy.Paginator(
            self.client.get_users_tweets,
            id=user_resp.data.id,
            tweet_fields=self.TWEET_FIELDS,
            expansions=["referenced_tweets.id"],
            max_results=min(max_results, 100),
        ).flatten(limit=max_results)

        for tweet in paginator:
            record = self._normalise_tweet(tweet)
            record["author"] = user
            tweets.append(record)

        logger.info("Collected %d tweets from @%s", len(tweets), username)
        return tweets

    def get_followers(self, user_id: str, max_results: int = 1000) -> list[dict]:
        """Return normalised user records for followers of *user_id*."""
        followers = []
        paginator = tweepy.Paginator(
            self.client.get_users_followers,
            id=user_id,
            user_fields=self.USER_FIELDS,
            max_results=min(max_results, 1000),
        ).flatten(limit=max_results)
        for user in paginator:
            followers.append(self._normalise_user(user))
        return followers

    def _fetch_users_by_ids(self, user_ids: list[str]) -> dict[str, dict]:
        """Bulk-fetch up to 100 users per request."""
        result: dict[str, dict] = {}
        for chunk_start in range(0, len(user_ids), 100):
            chunk = user_ids[chunk_start : chunk_start + 100]
            resp = self.client.get_users(
                ids=chunk,
                user_fields=self.USER_FIELDS,
            )
            if resp.data:
                for u in resp.data:
                    result[str(u.id)] = self._normalise_user(u)
            time.sleep(0.5)
        return result

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_tweet(tweet) -> dict:
        metrics = getattr(tweet, "public_metrics", {}) or {}
        return {
            "id": str(tweet.id),
            "text": tweet.text or "",
            "author_id": str(tweet.author_id) if tweet.author_id else None,
            "created_at": (
                tweet.created_at.isoformat()
                if getattr(tweet, "created_at", None)
                else None
            ),
            "lang": getattr(tweet, "lang", None),
            "retweet_count": metrics.get("retweet_count", 0),
            "reply_count": metrics.get("reply_count", 0),
            "like_count": metrics.get("like_count", 0),
            "quote_count": metrics.get("quote_count", 0),
            "referenced_tweets": [
                {"type": r.type, "id": str(r.id)}
                for r in (getattr(tweet, "referenced_tweets", None) or [])
            ],
            "entities": getattr(tweet, "entities", None) or {},
            "source": getattr(tweet, "source", None),
        }

    @staticmethod
    def _normalise_user(user) -> dict:
        metrics = getattr(user, "public_metrics", {}) or {}
        created = getattr(user, "created_at", None)
        if created:
            if hasattr(created, "isoformat"):
                created = created.isoformat()
            age_days = (datetime.now(timezone.utc) - user.created_at).days
        else:
            age_days = None

        return {
            "id": str(user.id),
            "name": user.name or "",
            "username": user.username or "",
            "created_at": created,
            "account_age_days": age_days,
            "description": getattr(user, "description", "") or "",
            "location": getattr(user, "location", "") or "",
            "profile_image_url": getattr(user, "profile_image_url", None),
            "verified": getattr(user, "verified", False) or False,
            "protected": getattr(user, "protected", False) or False,
            "followers_count": metrics.get("followers_count", 0),
            "following_count": metrics.get("following_count", 0),
            "tweet_count": metrics.get("tweet_count", 0),
            "listed_count": metrics.get("listed_count", 0),
        }
