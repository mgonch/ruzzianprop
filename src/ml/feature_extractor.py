"""
Feature extractor.

Converts raw account + tweet records into a fixed-width numerical feature
vector suitable for ML training and inference.

Feature groups
--------------
Account-level (10 features)
  account_age_days, followers_count, following_count, tweet_count,
  listed_count, follower_ratio, is_verified, has_bio, has_location, has_avatar

Behavioral (9 features)
  avg_tweets_per_day, retweet_ratio, reply_ratio, quote_ratio,
  posting_hour_entropy, posting_interval_cv, active_hours_span,
  username_digit_ratio, username_length

Content (6 features)
  avg_tweet_length, disinfo_keyword_density, state_media_link_ratio,
  avg_hashtags_per_tweet, avg_mentions_per_tweet, unique_text_ratio

Network (4 features – populated after graph analysis)
  degree_centrality, betweenness_centrality, pagerank, community_bot_ratio

Heuristic signals (10 features)
  The 10 raw signal scores from BotDetector (passed through as-is so the
  ML model can learn how to re-weight them).

Total: 39 features
"""

import math
import re
import logging
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HASHTAG_RE = re.compile(r"#\w+")
_MENTION_RE = re.compile(r"@\w+")

FEATURE_NAMES: list[str] = [
    # Account
    "account_age_days",
    "followers_count",
    "following_count",
    "tweet_count",
    "listed_count",
    "follower_ratio",
    "is_verified",
    "has_bio",
    "has_location",
    "has_avatar",
    # Behavioral
    "avg_tweets_per_day",
    "retweet_ratio",
    "reply_ratio",
    "quote_ratio",
    "posting_hour_entropy",
    "posting_interval_cv",
    "active_hours_span",
    "username_digit_ratio",
    "username_length",
    # Content
    "avg_tweet_length",
    "disinfo_keyword_density",
    "state_media_link_ratio",
    "avg_hashtags_per_tweet",
    "avg_mentions_per_tweet",
    "unique_text_ratio",
    # Network (may be 0 if graph not computed)
    "degree_centrality",
    "betweenness_centrality",
    "pagerank",
    "community_bot_ratio",
    # Heuristic signal pass-through
    "sig_account_age",
    "sig_username_pattern",
    "sig_profile_completeness",
    "sig_tweet_frequency",
    "sig_retweet_ratio",
    "sig_follower_ratio",
    "sig_content_similarity",
    "sig_posting_time_pattern",
    "sig_coordinated_behavior",
    "sig_disinformation_keywords",
]


class FeatureExtractor:
    """
    Extracts a fixed-width feature vector from account + tweet data.

    Parameters
    ----------
    keywords   : dict of {category: [keyword, ...]} for disinfo detection
    state_media: set of known Russian state media domains
    """

    def __init__(
        self,
        keywords: dict | None = None,
        state_media: set | None = None,
    ):
        self.keywords: list[str] = []
        for kw_list in (keywords or {}).values():
            self.keywords.extend([k.lower() for k in kw_list])

        self.state_media: set[str] = state_media or {
            "rt.com", "sputniknews.com", "ria.ru", "tass.ru",
            "rbth.com", "pravda.ru", "vesti.ru",
        }

    # ------------------------------------------------------------------

    def extract_one(
        self,
        user: dict,
        tweets: list[dict],
        heuristic_signals: dict | None = None,
        graph_attrs: dict | None = None,
    ) -> np.ndarray:
        """
        Return a 1-D float32 array of length len(FEATURE_NAMES).

        Parameters
        ----------
        user              : normalised user dict
        tweets            : list of normalised tweet dicts authored by this user
        heuristic_signals : raw signal scores from BotDetector.score_account()["signals"]
        graph_attrs       : dict with keys degree_centrality, betweenness_centrality,
                            pagerank, community_bot_ratio (all optional)
        """
        vec: dict[str, float] = {}

        # --- Account features ---
        age = float(user.get("account_age_days") or 0)
        followers = float(user.get("followers_count") or 0)
        following = float(user.get("following_count") or 0)
        tweet_count = float(user.get("tweet_count") or 0)
        listed = float(user.get("listed_count") or 0)

        vec["account_age_days"] = age
        vec["followers_count"] = self._log1p(followers)
        vec["following_count"] = self._log1p(following)
        vec["tweet_count"] = self._log1p(tweet_count)
        vec["listed_count"] = self._log1p(listed)
        vec["follower_ratio"] = following / max(followers, 1)
        vec["is_verified"] = float(bool(user.get("verified")))
        vec["has_bio"] = float(bool((user.get("description") or "").strip()))
        vec["has_location"] = float(bool((user.get("location") or "").strip()))
        img = user.get("profile_image_url") or ""
        vec["has_avatar"] = float(bool(img) and "default_profile" not in img)

        # --- Behavioral features ---
        timestamps = self._parse_timestamps(tweets)
        vec["avg_tweets_per_day"] = self._tweets_per_day(timestamps, age)
        vec["retweet_ratio"] = self._ratio(tweets, "retweeted")
        vec["reply_ratio"] = self._ratio(tweets, "replied_to")
        vec["quote_ratio"] = self._ratio(tweets, "quoted")
        vec["posting_hour_entropy"] = self._hour_entropy(timestamps)
        vec["posting_interval_cv"] = self._interval_cv(timestamps)
        vec["active_hours_span"] = self._active_hours_span(timestamps)

        username = user.get("username") or ""
        vec["username_digit_ratio"] = (
            sum(c.isdigit() for c in username) / max(len(username), 1)
        )
        vec["username_length"] = float(len(username))

        # --- Content features ---
        texts = [t.get("text") or "" for t in tweets]
        vec["avg_tweet_length"] = float(
            sum(len(t) for t in texts) / max(len(texts), 1)
        )
        vec["disinfo_keyword_density"] = self._disinfo_density(texts)
        vec["state_media_link_ratio"] = self._state_media_ratio(tweets)
        vec["avg_hashtags_per_tweet"] = float(
            sum(len(_HASHTAG_RE.findall(t)) for t in texts) / max(len(texts), 1)
        )
        vec["avg_mentions_per_tweet"] = float(
            sum(len(_MENTION_RE.findall(t)) for t in texts) / max(len(texts), 1)
        )
        # Uniqueness: how many distinct texts relative to total (dedup)
        unique_ratio = len(set(t[:100] for t in texts)) / max(len(texts), 1)
        vec["unique_text_ratio"] = unique_ratio

        # --- Network features (zeros if not available) ---
        ga = graph_attrs or {}
        vec["degree_centrality"] = float(ga.get("degree_centrality", 0.0))
        vec["betweenness_centrality"] = float(ga.get("betweenness_centrality", 0.0))
        vec["pagerank"] = float(ga.get("pagerank", 0.0))
        vec["community_bot_ratio"] = float(ga.get("community_bot_ratio", 0.0))

        # --- Heuristic signal pass-through ---
        sig = heuristic_signals or {}
        for signal_key in [
            "account_age", "username_pattern", "profile_completeness",
            "tweet_frequency", "retweet_ratio", "follower_ratio",
            "content_similarity", "posting_time_pattern",
            "coordinated_behavior", "disinformation_keywords",
        ]:
            vec[f"sig_{signal_key}"] = float(sig.get(signal_key, 0.0))

        return np.array([vec[f] for f in FEATURE_NAMES], dtype=np.float32)

    def extract_batch(
        self,
        records: list[dict],
        heuristic_scores: list[dict] | None = None,
        graph: object = None,  # nx.DiGraph or None
    ) -> pd.DataFrame:
        """
        Extract features for every unique author in *records*.

        Returns a DataFrame with columns = FEATURE_NAMES and
        index = user_id strings.
        """
        from collections import defaultdict

        by_author: dict[str, dict] = {}
        tweets_by_author: dict[str, list[dict]] = defaultdict(list)

        for r in records:
            author = r.get("author") or {}
            uid = author.get("id") or r.get("author_id", "")
            if uid and uid not in by_author:
                by_author[uid] = author
            if uid:
                tweets_by_author[uid].append(r)

        # Index heuristic scores
        score_map: dict[str, dict] = {}
        if heuristic_scores:
            for s in heuristic_scores:
                uid = s.get("user_id") or (s.get("author") or {}).get("id")
                if uid:
                    score_map[uid] = s.get("signals", {})

        # Index graph attributes
        graph_attr_map: dict[str, dict] = {}
        if graph is not None:
            try:
                import networkx as nx
                for node in graph.nodes():
                    attrs = graph.nodes[node]
                    community_id = attrs.get("community_id")
                    # Compute community bot ratio
                    if community_id is not None:
                        peers = [
                            n for n in graph.nodes()
                            if graph.nodes[n].get("community_id") == community_id
                        ]
                        if peers:
                            bot_ratio = sum(
                                1 for p in peers
                                if graph.nodes[p].get("classification") == "bot"
                            ) / len(peers)
                        else:
                            bot_ratio = 0.0
                    else:
                        bot_ratio = 0.0

                    graph_attr_map[str(node)] = {
                        "degree_centrality": attrs.get("degree_centrality", 0.0),
                        "betweenness_centrality": attrs.get("betweenness_centrality", 0.0),
                        "pagerank": attrs.get("pagerank", 0.0),
                        "community_bot_ratio": bot_ratio,
                    }
            except Exception as e:
                logger.warning("Could not extract graph attributes: %s", e)

        rows = []
        indices = []
        for uid, user in by_author.items():
            vec = self.extract_one(
                user,
                tweets_by_author.get(uid, []),
                heuristic_signals=score_map.get(uid),
                graph_attrs=graph_attr_map.get(uid),
            )
            rows.append(vec)
            indices.append(uid)

        if not rows:
            return pd.DataFrame(columns=FEATURE_NAMES)

        df = pd.DataFrame(rows, columns=FEATURE_NAMES, index=indices)
        logger.info("Extracted features for %d accounts (%d features each)", len(df), len(FEATURE_NAMES))
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log1p(x: float) -> float:
        return float(np.log1p(max(x, 0)))

    @staticmethod
    def _parse_timestamps(tweets: list[dict]) -> list[datetime]:
        result = []
        for t in tweets:
            ts = t.get("created_at")
            if ts:
                try:
                    result.append(
                        datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    )
                except ValueError:
                    pass
        return sorted(result)

    @staticmethod
    def _tweets_per_day(timestamps: list[datetime], age_days: float) -> float:
        if timestamps and age_days > 0:
            span = max(
                (timestamps[-1] - timestamps[0]).total_seconds() / 86400
                if len(timestamps) > 1 else 1,
                1,
            )
            return min(len(timestamps) / span, 500)
        return 0.0

    @staticmethod
    def _ratio(tweets: list[dict], ref_type: str) -> float:
        if not tweets:
            return 0.0
        count = sum(
            1 for t in tweets
            if any(r.get("type") == ref_type for r in (t.get("referenced_tweets") or []))
        )
        return count / len(tweets)

    @staticmethod
    def _hour_entropy(timestamps: list[datetime]) -> float:
        """Shannon entropy of posting hours. Max entropy = log2(24) ≈ 4.58."""
        if len(timestamps) < 3:
            return 0.0
        counts = Counter(dt.hour for dt in timestamps)
        total = sum(counts.values())
        entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in counts.values() if c > 0
        )
        return entropy / math.log2(24)  # Normalise to [0, 1]

    @staticmethod
    def _interval_cv(timestamps: list[datetime]) -> float:
        """Coefficient of variation of inter-tweet intervals. Low CV = robotic."""
        if len(timestamps) < 3:
            return 1.0  # Unknown → assume natural
        gaps = [
            (timestamps[i + 1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]
        mean = sum(gaps) / len(gaps)
        if mean <= 0:
            return 0.0
        std = math.sqrt(sum((g - mean) ** 2 for g in gaps) / len(gaps))
        return min(std / mean, 10.0)  # Cap at 10

    @staticmethod
    def _active_hours_span(timestamps: list[datetime]) -> float:
        """Fraction of the 24 hours in which the account was active."""
        if not timestamps:
            return 0.0
        return len(set(dt.hour for dt in timestamps)) / 24.0

    def _disinfo_density(self, texts: list[str]) -> float:
        if not texts or not self.keywords:
            return 0.0
        hits = sum(
            1 for text in texts
            if any(kw in text.lower() for kw in self.keywords)
        )
        return hits / len(texts)

    def _state_media_ratio(self, tweets: list[dict]) -> float:
        if not tweets:
            return 0.0
        from urllib.parse import urlparse
        hits = 0
        for t in tweets:
            entities = t.get("entities") or {}
            for url_obj in entities.get("urls") or []:
                expanded = url_obj.get("expanded_url") or url_obj.get("url") or ""
                try:
                    domain = urlparse(expanded).netloc.lower().lstrip("www.")
                    if any(
                        domain == sm or domain.endswith("." + sm)
                        for sm in self.state_media
                    ):
                        hits += 1
                        break  # Count once per tweet
                except Exception:
                    pass
        return hits / len(tweets)
