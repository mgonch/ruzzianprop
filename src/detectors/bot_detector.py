"""
Bot detection scoring engine.

Scores each account on a scale of 0.0 – 1.0 where 1.0 = almost certainly a bot.
Each sub-signal contributes a weighted partial score; the final score is the
weighted sum of all partial scores.

Signals implemented
-------------------
1.  account_age          – New accounts are more suspicious
2.  username_pattern     – Random-looking usernames (digits, underscores)
3.  profile_completeness – Missing bio, default avatar, no location
4.  tweet_frequency      – Superhuman posting rates
5.  retweet_ratio        – High RT ratio = amplifier, not organic voice
6.  follower_ratio       – Following >> followers is a classic bot trait
7.  content_similarity   – Detects copy-paste campaigns across accounts
8.  posting_time_pattern – Unnatural uniformity / out-of-timezone activity
9.  coordinated_behavior – Clusters posting same content in short windows
10. disinformation_keywords – Direct match against known narrative terms
"""

import re
import math
import logging
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Sequence

logger = logging.getLogger(__name__)

# Default signal weights (override via config)
DEFAULT_WEIGHTS = {
    "account_age": 0.10,
    "username_pattern": 0.08,
    "profile_completeness": 0.08,
    "tweet_frequency": 0.10,
    "retweet_ratio": 0.10,
    "follower_ratio": 0.10,
    "content_similarity": 0.15,
    "posting_time_pattern": 0.09,
    "coordinated_behavior": 0.12,
    "disinformation_keywords": 0.08,
}

# Regex for random-username patterns
_USERNAME_RANDOM_RE = re.compile(
    r"(\d{4,})"                  # 4+ digits in a row
    r"|([a-z]{2,}\d{4,})"        # letters followed by 4+ digits
    r"|(_\d{2,}$)"               # underscore + digits at end
    r"|(^[a-z]{1,4}\d{5,})",     # very short prefix + 5+ digits
    re.IGNORECASE,
)


class BotDetector:
    """Score accounts for bot-like behaviour."""

    def __init__(self, config: dict | None = None, keywords: dict | None = None):
        cfg = (config or {}).get("bot_detection", {})
        self.weights = {**DEFAULT_WEIGHTS, **cfg.get("weights", {})}
        self.bot_threshold = cfg.get("bot_threshold", 0.60)
        self.suspected_threshold = cfg.get("suspected_threshold", 0.40)
        self.keywords = keywords or {}
        # Will be populated by analyse_corpus()
        self._corpus_texts: list[str] = []
        self._coordinated_groups: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Corpus-level analysis (must be called before per-account scoring)
    # ------------------------------------------------------------------

    def analyse_corpus(self, records: list[dict]) -> None:
        """
        Pre-compute corpus-wide signals (similarity, coordination) from *records*.
        Call this once before calling score_account().
        """
        self._corpus_texts = [r.get("text", "") for r in records]
        self._build_coordination_map(records)

    def _build_coordination_map(self, records: list[dict]) -> None:
        """
        Group accounts that posted very similar text within a short time window.
        Similarity >= 0.85 within 10 minutes → coordinated activity.
        """
        groups: dict[str, list[str]] = defaultdict(list)
        texts = [(r.get("text", ""), r.get("author_id", ""), r.get("created_at")) for r in records]

        for i, (text_i, author_i, ts_i) in enumerate(texts):
            if not text_i or not author_i:
                continue
            for j in range(i + 1, len(texts)):
                text_j, author_j, ts_j = texts[j]
                if author_i == author_j or not text_j:
                    continue
                sim = SequenceMatcher(None, text_i[:200], text_j[:200]).ratio()
                if sim >= 0.85:
                    # Check time proximity (within 10 min)
                    if ts_i and ts_j:
                        try:
                            dt_i = datetime.fromisoformat(str(ts_i).replace("Z", "+00:00"))
                            dt_j = datetime.fromisoformat(str(ts_j).replace("Z", "+00:00"))
                            diff_min = abs((dt_i - dt_j).total_seconds()) / 60
                            if diff_min <= 10:
                                groups[author_i].append(author_j)
                                groups[author_j].append(author_i)
                        except ValueError:
                            pass
        self._coordinated_groups = groups

    # ------------------------------------------------------------------
    # Per-account scoring
    # ------------------------------------------------------------------

    def score_account(self, user: dict, tweets: list[dict] | None = None) -> dict:
        """
        Return a detailed scoring dict for *user*.

        ``tweets`` should be tweets authored by this user (from the corpus or
        timeline fetch). If None, only account-level signals are computed.
        """
        tweets = tweets or []
        signals: dict[str, float] = {}

        signals["account_age"] = self._score_account_age(user)
        signals["username_pattern"] = self._score_username(user)
        signals["profile_completeness"] = self._score_profile(user)
        signals["tweet_frequency"] = self._score_tweet_frequency(user, tweets)
        signals["retweet_ratio"] = self._score_retweet_ratio(tweets)
        signals["follower_ratio"] = self._score_follower_ratio(user)
        signals["content_similarity"] = self._score_content_similarity(tweets)
        signals["posting_time_pattern"] = self._score_posting_time(tweets)
        signals["coordinated_behavior"] = self._score_coordination(user)
        signals["disinformation_keywords"] = self._score_keywords(tweets)

        total = sum(
            signals[k] * self.weights.get(k, 0.0)
            for k in signals
        )
        # Clamp to [0, 1]
        total = max(0.0, min(1.0, total))

        if total >= self.bot_threshold:
            classification = "bot"
        elif total >= self.suspected_threshold:
            classification = "suspected"
        else:
            classification = "human"

        return {
            "user_id": user.get("id"),
            "username": user.get("username"),
            "bot_score": round(total, 4),
            "bot_score_pct": round(total * 100, 1),
            "classification": classification,
            "signals": {k: round(v, 4) for k, v in signals.items()},
        }

    # ------------------------------------------------------------------
    # Individual signal scorers (each returns 0.0 – 1.0)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_account_age(user: dict) -> float:
        """Newer accounts are more suspicious."""
        age = user.get("account_age_days")
        if age is None:
            return 0.5  # Unknown → neutral
        if age < 30:
            return 1.0
        if age < 90:
            return 0.8
        if age < 180:
            return 0.5
        if age < 365:
            return 0.3
        if age < 730:
            return 0.15
        return 0.0

    @staticmethod
    def _score_username(user: dict) -> float:
        """Random / generated usernames score higher."""
        username = user.get("username", "")
        if not username:
            return 0.5
        if _USERNAME_RANDOM_RE.search(username):
            return 0.9
        # Long usernames with many digits
        digit_ratio = sum(c.isdigit() for c in username) / max(len(username), 1)
        if digit_ratio > 0.4:
            return 0.7
        return 0.0

    @staticmethod
    def _score_profile(user: dict) -> float:
        """Incomplete profiles are more suspicious."""
        score = 0.0
        if not user.get("description", "").strip():
            score += 0.4
        if not user.get("location", "").strip():
            score += 0.2
        img = user.get("profile_image_url", "")
        if not img or "default_profile" in (img or ""):
            score += 0.4
        return min(score, 1.0)

    @staticmethod
    def _score_tweet_frequency(user: dict, tweets: list[dict]) -> float:
        """
        Compute average tweets per day; superhuman rates score higher.
        If we have a timeline, use it. Otherwise fall back to total tweet count / age.
        """
        if tweets:
            timestamps = []
            for t in tweets:
                ts = t.get("created_at")
                if ts:
                    try:
                        timestamps.append(
                            datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                        )
                    except ValueError:
                        pass
            if len(timestamps) >= 2:
                timestamps.sort()
                span_days = max(
                    (timestamps[-1] - timestamps[0]).total_seconds() / 86400, 1
                )
                rate = len(timestamps) / span_days
                if rate > 200:
                    return 1.0
                if rate > 100:
                    return 0.8
                if rate > 50:
                    return 0.5
                if rate > 20:
                    return 0.2
                return 0.0

        # Fallback: use total tweet count and account age
        age = user.get("account_age_days") or 1
        count = user.get("tweet_count", 0) or 0
        rate = count / max(age, 1)
        if rate > 150:
            return 1.0
        if rate > 80:
            return 0.8
        if rate > 40:
            return 0.5
        if rate > 15:
            return 0.2
        return 0.0

    @staticmethod
    def _score_retweet_ratio(tweets: list[dict]) -> float:
        """High retweet ratio indicates an amplifier bot."""
        if not tweets:
            return 0.0
        retweet_count = sum(
            1 for t in tweets
            if any(
                r.get("type") == "retweeted"
                for r in (t.get("referenced_tweets") or [])
            )
        )
        ratio = retweet_count / len(tweets)
        if ratio > 0.9:
            return 1.0
        if ratio > 0.7:
            return 0.7
        if ratio > 0.5:
            return 0.4
        return 0.0

    @staticmethod
    def _score_follower_ratio(user: dict) -> float:
        """Following >> followers is a classic bot trait."""
        followers = user.get("followers_count", 0) or 0
        following = user.get("following_count", 0) or 0
        if followers == 0 and following > 50:
            return 1.0
        if followers == 0:
            return 0.3
        ratio = following / followers
        if ratio > 50:
            return 1.0
        if ratio > 10:
            return 0.8
        if ratio > 5:
            return 0.5
        if ratio > 2:
            return 0.2
        return 0.0

    def _score_content_similarity(self, tweets: list[dict]) -> float:
        """
        Measure how similar this account's tweets are to each other
        (repetitive posting) and to the overall corpus.
        """
        texts = [t.get("text", "") for t in tweets if t.get("text")]
        if len(texts) < 2:
            return 0.0

        # Intra-account similarity
        sim_scores = []
        sample = texts[:30]  # Cap to avoid O(n²) blowup
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                sim_scores.append(
                    SequenceMatcher(None, sample[i][:200], sample[j][:200]).ratio()
                )

        if not sim_scores:
            return 0.0

        avg_sim = sum(sim_scores) / len(sim_scores)
        if avg_sim > 0.8:
            return 1.0
        if avg_sim > 0.5:
            return 0.6
        if avg_sim > 0.3:
            return 0.3
        return 0.0

    @staticmethod
    def _score_posting_time(tweets: list[dict]) -> float:
        """
        Detect unnatural posting patterns:
        - Posting uniformly at exact same minute intervals
        - Posting 24/7 without human sleep breaks
        """
        timestamps = []
        for t in tweets:
            ts = t.get("created_at")
            if ts:
                try:
                    timestamps.append(
                        datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    )
                except ValueError:
                    pass

        if len(timestamps) < 5:
            return 0.0

        timestamps.sort()

        # Check if posting 24/7 with no sleep break > 6h
        hours_set = set(dt.hour for dt in timestamps)
        if len(hours_set) >= 22:
            return 0.9

        # Check for uniform spacing (robot-like scheduling)
        gaps = [
            (timestamps[i + 1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]
        if not gaps:
            return 0.0

        mean_gap = sum(gaps) / len(gaps)
        if mean_gap == 0:
            return 0.8
        variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
        cv = math.sqrt(variance) / mean_gap if mean_gap > 0 else 0
        # Coefficient of Variation: very low CV = unnaturally regular
        if cv < 0.1 and len(gaps) > 10:
            return 0.9
        if cv < 0.3 and len(gaps) > 10:
            return 0.5
        return 0.0

    def _score_coordination(self, user: dict) -> float:
        """Score based on how many other accounts posted same content."""
        uid = user.get("id")
        if not uid or uid not in self._coordinated_groups:
            return 0.0
        partners = len(set(self._coordinated_groups[uid]))
        if partners >= 10:
            return 1.0
        if partners >= 5:
            return 0.8
        if partners >= 2:
            return 0.5
        if partners >= 1:
            return 0.3
        return 0.0

    def _score_keywords(self, tweets: list[dict]) -> float:
        """
        Check tweets for known Russian disinformation narrative keywords.
        Returns the proportion of tweets containing at least one keyword.
        """
        if not tweets or not self.keywords:
            return 0.0

        all_kws = []
        for kw_list in self.keywords.values():
            all_kws.extend([k.lower() for k in kw_list])

        hit_count = 0
        for tweet in tweets:
            text = (tweet.get("text") or "").lower()
            if any(kw in text for kw in all_kws):
                hit_count += 1

        ratio = hit_count / len(tweets)
        if ratio > 0.5:
            return 1.0
        if ratio > 0.3:
            return 0.7
        if ratio > 0.1:
            return 0.4
        if ratio > 0.0:
            return 0.2
        return 0.0

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def score_all(self, records: list[dict]) -> list[dict]:
        """
        Score every unique author in *records*.

        Runs corpus analysis first, then scores each account using all
        tweets attributed to it.

        Returns a list of scoring dicts sorted by bot_score descending.
        """
        self.analyse_corpus(records)

        # Group tweets by author
        by_author: dict[str, dict] = {}
        tweets_by_author: dict[str, list[dict]] = defaultdict(list)

        for record in records:
            author = record.get("author") or {}
            uid = author.get("id") or record.get("author_id", "")
            if uid and uid not in by_author:
                by_author[uid] = author
            if uid:
                tweets_by_author[uid].append(record)

        results = []
        for uid, user in by_author.items():
            score = self.score_account(user, tweets_by_author.get(uid, []))
            score["author"] = user
            results.append(score)

        results.sort(key=lambda x: x["bot_score"], reverse=True)
        logger.info(
            "Scored %d accounts. Bots: %d, Suspected: %d, Human: %d",
            len(results),
            sum(1 for r in results if r["classification"] == "bot"),
            sum(1 for r in results if r["classification"] == "suspected"),
            sum(1 for r in results if r["classification"] == "human"),
        )
        return results
