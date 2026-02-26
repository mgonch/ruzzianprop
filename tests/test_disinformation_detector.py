"""Tests for the DisinformationDetector."""

import pytest
from src.detectors.disinformation_detector import DisinformationDetector


SAMPLE_CONFIG = {
    "disinformation": {
        "keywords": {
            "nato_narratives": ["nato aggression", "nato provocation"],
            "ukraine_narratives": ["denazification", "zelensky nazi", "kiev regime"],
            "election_narratives": ["stolen election", "deep state"],
        },
        "state_media_domains": ["rt.com", "sputniknews.com"],
    }
}


def make_tweet(text: str, urls: list | None = None) -> dict:
    return {
        "id": "1",
        "text": text,
        "entities": {"urls": urls or []},
    }


class TestAnalyseTweet:
    def test_disinfo_keyword_detected(self):
        det = DisinformationDetector(SAMPLE_CONFIG)
        result = det.analyse_tweet(make_tweet("NATO aggression must be condemned"))
        assert "nato_narratives" in result["narratives"]
        assert result["dis_score"] > 0

    def test_clean_tweet_zero_score(self):
        det = DisinformationDetector(SAMPLE_CONFIG)
        result = det.analyse_tweet(make_tweet("Just had coffee with a friend!"))
        assert result["dis_score"] == 0.0
        assert result["narratives"] == []

    def test_state_media_link_detected(self):
        det = DisinformationDetector(SAMPLE_CONFIG)
        tweet = make_tweet(
            "Check this out",
            urls=[{"expanded_url": "https://rt.com/news/some-article"}],
        )
        result = det.analyse_tweet(tweet)
        assert len(result["state_media_links"]) > 0
        assert result["dis_score"] > 0

    def test_multiple_narratives(self):
        det = DisinformationDetector(SAMPLE_CONFIG)
        result = det.analyse_tweet(
            make_tweet("NATO aggression and denazification and stolen election")
        )
        assert len(result["narratives"]) == 3

    def test_case_insensitive(self):
        det = DisinformationDetector(SAMPLE_CONFIG)
        result = det.analyse_tweet(make_tweet("NATO AGGRESSION IS WRONG"))
        assert "nato_narratives" in result["narratives"]


class TestAnalyseAccount:
    def test_empty_tweets_returns_zeros(self):
        det = DisinformationDetector(SAMPLE_CONFIG)
        result = det.analyse_account([])
        assert result["avg_dis_score"] == 0.0
        assert result["dis_tweet_ratio"] == 0.0

    def test_high_dis_account(self):
        det = DisinformationDetector(SAMPLE_CONFIG)
        tweets = [
            make_tweet("NATO aggression everywhere!"),
            make_tweet("Kiev regime must fall. Zelensky nazi"),
            make_tweet("Stolen election! Deep state is real"),
        ]
        result = det.analyse_account(tweets)
        assert result["avg_dis_score"] > 0.2
        assert result["dis_tweet_ratio"] == 1.0

    def test_mixed_account(self):
        det = DisinformationDetector(SAMPLE_CONFIG)
        tweets = [
            make_tweet("NATO aggression!"),
            make_tweet("I love coffee"),
            make_tweet("Great weather today"),
        ]
        result = det.analyse_account(tweets)
        assert 0 < result["dis_tweet_ratio"] < 1.0


class TestAnalyseCorpus:
    def test_corpus_returns_top_narratives(self):
        det = DisinformationDetector(SAMPLE_CONFIG)
        records = [
            {
                "text": "NATO aggression",
                "entities": {},
                "author": {"id": "u1"},
                "author_id": "u1",
            },
            {
                "text": "Deep state everywhere",
                "entities": {},
                "author": {"id": "u2"},
                "author_id": "u2",
            },
        ]
        result = det.analyse_corpus(records)
        assert result["total_tweets"] == 2
        assert result["total_accounts"] == 2
        assert len(result["top_narratives"]) > 0
