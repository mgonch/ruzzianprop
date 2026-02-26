"""Tests for the BotDetector scoring engine."""

import pytest
from src.detectors.bot_detector import BotDetector


def make_user(**kwargs) -> dict:
    defaults = {
        "id": "123",
        "username": "normaluser",
        "name": "Normal User",
        "created_at": "2018-01-01T00:00:00+00:00",
        "account_age_days": 1000,
        "description": "Regular person tweeting about stuff.",
        "location": "New York",
        "profile_image_url": "https://example.com/photo.jpg",
        "verified": False,
        "protected": False,
        "followers_count": 500,
        "following_count": 300,
        "tweet_count": 2000,
        "listed_count": 10,
    }
    defaults.update(kwargs)
    return defaults


def make_tweet(text: str = "Hello world", created_at: str = "2024-01-01T12:00:00+00:00",
               is_retweet: bool = False) -> dict:
    return {
        "id": "456",
        "text": ("RT @other: " + text) if is_retweet else text,
        "author_id": "123",
        "created_at": created_at,
        "referenced_tweets": [{"type": "retweeted", "id": "789"}] if is_retweet else [],
        "entities": {},
    }


class TestAccountAge:
    def test_new_account_scores_high(self):
        det = BotDetector()
        user = make_user(account_age_days=10)
        result = det._score_account_age(user)
        assert result == 1.0

    def test_old_account_scores_low(self):
        det = BotDetector()
        user = make_user(account_age_days=1000)
        result = det._score_account_age(user)
        assert result == 0.0

    def test_unknown_age_neutral(self):
        det = BotDetector()
        user = make_user(account_age_days=None)
        result = det._score_account_age(user)
        assert result == 0.5


class TestUsernamePattern:
    def test_random_digits_detected(self):
        det = BotDetector()
        assert det._score_username(make_user(username="user12345678")) > 0.5

    def test_human_username_low_score(self):
        det = BotDetector()
        assert det._score_username(make_user(username="alice_johnson")) == 0.0

    def test_pure_digits_detected(self):
        det = BotDetector()
        assert det._score_username(make_user(username="abc9999999")) > 0.5


class TestProfileCompleteness:
    def test_empty_profile_scores_high(self):
        det = BotDetector()
        user = make_user(description="", location="", profile_image_url=None)
        assert det._score_profile(user) >= 0.8

    def test_complete_profile_scores_low(self):
        det = BotDetector()
        user = make_user(
            description="Software engineer",
            location="London",
            profile_image_url="https://example.com/photo.jpg",
        )
        assert det._score_profile(user) == 0.0


class TestRetweetRatio:
    def test_all_retweets_scores_high(self):
        det = BotDetector()
        tweets = [make_tweet(is_retweet=True) for _ in range(10)]
        assert det._score_retweet_ratio(tweets) == 1.0

    def test_no_retweets_scores_zero(self):
        det = BotDetector()
        tweets = [make_tweet(is_retweet=False) for _ in range(10)]
        assert det._score_retweet_ratio(tweets) == 0.0

    def test_empty_tweets_scores_zero(self):
        det = BotDetector()
        assert det._score_retweet_ratio([]) == 0.0


class TestFollowerRatio:
    def test_zero_followers_many_following(self):
        det = BotDetector()
        user = make_user(followers_count=0, following_count=500)
        assert det._score_follower_ratio(user) == 1.0

    def test_normal_ratio_low_score(self):
        det = BotDetector()
        user = make_user(followers_count=1000, following_count=500)
        assert det._score_follower_ratio(user) == 0.0

    def test_extreme_ratio_high_score(self):
        det = BotDetector()
        user = make_user(followers_count=10, following_count=5000)
        assert det._score_follower_ratio(user) >= 0.8


class TestKeywords:
    def test_disinfo_keywords_detected(self):
        det = BotDetector(keywords={"test": ["nato aggression", "denazification"]})
        tweets = [make_tweet("NATO aggression is wrong"), make_tweet("Hello world")]
        score = det._score_keywords(tweets)
        assert score > 0.0

    def test_clean_text_zero_score(self):
        det = BotDetector(keywords={"test": ["nato aggression"]})
        tweets = [make_tweet("I love coffee"), make_tweet("Great weather today")]
        assert det._score_keywords(tweets) == 0.0

    def test_empty_keywords_zero_score(self):
        det = BotDetector(keywords={})
        tweets = [make_tweet("NATO aggression")]
        assert det._score_keywords(tweets) == 0.0


class TestScoreAll:
    def test_score_all_returns_sorted_results(self):
        det = BotDetector()
        # One obvious bot, one obvious human
        records = [
            {
                "text": "Hello world",
                "author_id": "human_1",
                "created_at": "2024-01-01T12:00:00+00:00",
                "referenced_tweets": [],
                "entities": {},
                "author": make_user(
                    id="human_1", username="alice_smith",
                    account_age_days=1500, description="Normal person",
                    location="NYC", profile_image_url="https://pic.jpg",
                    followers_count=800, following_count=400,
                ),
            },
            {
                "text": "RT @someone: NATO aggression!",
                "author_id": "bot_1",
                "created_at": "2024-01-01T12:01:00+00:00",
                "referenced_tweets": [{"type": "retweeted", "id": "0"}],
                "entities": {},
                "author": make_user(
                    id="bot_1", username="user99887766",
                    account_age_days=5, description="",
                    location="", profile_image_url=None,
                    followers_count=0, following_count=3000,
                    tweet_count=50000,
                ),
            },
        ]
        results = det.score_all(records)
        assert len(results) == 2
        # Bot should score higher than human
        bot_result = next(r for r in results if r["user_id"] == "bot_1")
        human_result = next(r for r in results if r["user_id"] == "human_1")
        assert bot_result["bot_score"] > human_result["bot_score"]

    def test_score_all_empty(self):
        det = BotDetector()
        results = det.score_all([])
        assert results == []
