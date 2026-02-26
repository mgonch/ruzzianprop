"""
Disinformation content detector.

Analyses tweet text and linked domains for known Russian state-linked
disinformation narratives and state media sources.

Narrative categories are sourced from academic research on IRA (Internet
Research Agency) and similar Kremlin-linked influence operations:
- EU DisinfoLab reports
- Stanford Internet Observatory findings
- DFRLab (Digital Forensic Research Lab) databases
"""

import re
import logging
from collections import Counter
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DisinformationDetector:
    """Detect disinformation narratives in tweet content."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("disinformation", {})
        raw_keywords = cfg.get("keywords", {})
        self.keywords: dict[str, list[str]] = {
            cat: [k.lower() for k in kws]
            for cat, kws in raw_keywords.items()
        }
        self.state_media_domains: set[str] = set(
            cfg.get("state_media_domains", [
                "rt.com", "sputniknews.com", "ria.ru", "tass.ru",
                "rbth.com", "pravda.ru", "vesti.ru",
            ])
        )

    # ------------------------------------------------------------------

    def analyse_tweet(self, tweet: dict) -> dict:
        """
        Return a disinformation analysis for a single tweet.

        Returns a dict with:
          - dis_score   (0.0 – 1.0)
          - narratives  (list of matched narrative categories)
          - matched_keywords (list of matched keyword strings)
          - state_media_links (list of matched state media domains)
        """
        text = (tweet.get("text") or "").lower()
        entities = tweet.get("entities") or {}
        urls = entities.get("urls") or []

        matched_keywords: list[str] = []
        narratives: list[str] = []

        for category, kws in self.keywords.items():
            category_hits = [kw for kw in kws if kw in text]
            if category_hits:
                narratives.append(category)
                matched_keywords.extend(category_hits)

        # State media link detection
        state_media_links: list[str] = []
        for url_obj in urls:
            expanded = url_obj.get("expanded_url") or url_obj.get("url") or ""
            try:
                domain = urlparse(expanded).netloc.lower().lstrip("www.")
                if any(domain == sm or domain.endswith("." + sm) for sm in self.state_media_domains):
                    state_media_links.append(domain)
            except Exception:
                pass

        # Compute score
        kw_score = min(len(matched_keywords) / 5, 1.0) * 0.6
        narrative_score = min(len(narratives) / 3, 1.0) * 0.2
        media_score = min(len(state_media_links), 1) * 0.2

        dis_score = kw_score + narrative_score + media_score

        return {
            "tweet_id": tweet.get("id"),
            "dis_score": round(dis_score, 4),
            "narratives": narratives,
            "matched_keywords": list(set(matched_keywords)),
            "state_media_links": state_media_links,
        }

    def analyse_account(self, tweets: list[dict]) -> dict:
        """
        Aggregate disinformation signals across all of an account's tweets.
        Returns a summary dict.
        """
        if not tweets:
            return {
                "avg_dis_score": 0.0,
                "top_narratives": [],
                "top_keywords": [],
                "state_media_share": 0.0,
                "dis_tweet_count": 0,
                "dis_tweet_ratio": 0.0,
            }

        results = [self.analyse_tweet(t) for t in tweets]
        avg_score = sum(r["dis_score"] for r in results) / len(results)

        all_narratives = [n for r in results for n in r["narratives"]]
        all_keywords = [k for r in results for k in r["matched_keywords"]]
        all_media = [m for r in results for m in r["state_media_links"]]

        dis_tweets = sum(1 for r in results if r["dis_score"] > 0)

        top_narratives = [n for n, _ in Counter(all_narratives).most_common(5)]
        top_keywords = [k for k, _ in Counter(all_keywords).most_common(10)]

        state_media_share = (
            len([r for r in results if r["state_media_links"]]) / len(results)
        )

        return {
            "avg_dis_score": round(avg_score, 4),
            "top_narratives": top_narratives,
            "top_keywords": top_keywords,
            "state_media_share": round(state_media_share, 4),
            "dis_tweet_count": dis_tweets,
            "dis_tweet_ratio": round(dis_tweets / len(tweets), 4),
        }

    def analyse_corpus(self, records: list[dict]) -> dict:
        """
        Summarise disinformation signals across an entire dataset.
        Returns top narratives, keywords, and most active accounts.
        """
        from collections import defaultdict

        by_author: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            uid = (r.get("author") or {}).get("id") or r.get("author_id", "anonymous")
            by_author[uid].append(r)

        all_narratives: list[str] = []
        all_keywords: list[str] = []
        account_scores: list[tuple[str, float]] = []

        for uid, tweets in by_author.items():
            summary = self.analyse_account(tweets)
            all_narratives.extend(summary["top_narratives"])
            all_keywords.extend(summary["top_keywords"])
            if summary["avg_dis_score"] > 0:
                account_scores.append((uid, summary["avg_dis_score"]))

        account_scores.sort(key=lambda x: x[1], reverse=True)

        return {
            "total_tweets": len(records),
            "total_accounts": len(by_author),
            "top_narratives": [
                {"narrative": n, "count": c}
                for n, c in Counter(all_narratives).most_common(10)
            ],
            "top_keywords": [
                {"keyword": k, "count": c}
                for k, c in Counter(all_keywords).most_common(20)
            ],
            "most_active_disinfo_accounts": [
                {"user_id": uid, "avg_dis_score": score}
                for uid, score in account_scores[:20]
            ],
        }
