"""Tests for the ML FeatureExtractor."""

import numpy as np
import pytest
from src.ml.feature_extractor import FeatureExtractor, FEATURE_NAMES


def make_user(**kwargs) -> dict:
    defaults = {
        "id": "u1", "username": "alice", "name": "Alice",
        "account_age_days": 365, "description": "Bio here",
        "location": "NYC", "profile_image_url": "https://pic.jpg",
        "verified": False, "protected": False,
        "followers_count": 500, "following_count": 200,
        "tweet_count": 1000, "listed_count": 5,
    }
    defaults.update(kwargs)
    return defaults


def make_tweet(text="Hello", created_at="2024-06-01T12:00:00+00:00",
               is_rt=False) -> dict:
    return {
        "id": "t1", "text": text, "author_id": "u1",
        "created_at": created_at,
        "referenced_tweets": [{"type": "retweeted", "id": "0"}] if is_rt else [],
        "entities": {},
    }


class TestFeatureExtractor:
    def test_output_shape(self):
        ext = FeatureExtractor()
        user = make_user()
        tweets = [make_tweet() for _ in range(5)]
        vec = ext.extract_one(user, tweets)
        assert vec.shape == (len(FEATURE_NAMES),)
        assert vec.dtype == np.float32

    def test_feature_names_length(self):
        assert len(FEATURE_NAMES) == 39

    def test_no_tweets(self):
        ext = FeatureExtractor()
        vec = ext.extract_one(make_user(), [])
        assert vec.shape == (len(FEATURE_NAMES),)
        assert not np.any(np.isnan(vec))

    def test_verified_flag(self):
        ext = FeatureExtractor()
        verified = ext.extract_one(make_user(verified=True), [])
        unverified = ext.extract_one(make_user(verified=False), [])
        v_idx = FEATURE_NAMES.index("is_verified")
        assert verified[v_idx] == 1.0
        assert unverified[v_idx] == 0.0

    def test_empty_profile_detected(self):
        ext = FeatureExtractor()
        empty = ext.extract_one(
            make_user(description="", location="", profile_image_url=None), []
        )
        complete = ext.extract_one(make_user(), [])
        bio_idx = FEATURE_NAMES.index("has_bio")
        loc_idx = FEATURE_NAMES.index("has_location")
        assert empty[bio_idx] == 0.0
        assert complete[bio_idx] == 1.0
        assert empty[loc_idx] == 0.0

    def test_retweet_ratio(self):
        ext = FeatureExtractor()
        tweets_rt = [make_tweet(is_rt=True) for _ in range(5)]
        tweets_org = [make_tweet(is_rt=False) for _ in range(5)]
        rt_vec = ext.extract_one(make_user(), tweets_rt)
        org_vec = ext.extract_one(make_user(), tweets_org)
        rt_idx = FEATURE_NAMES.index("retweet_ratio")
        assert rt_vec[rt_idx] == pytest.approx(1.0)
        assert org_vec[rt_idx] == pytest.approx(0.0)

    def test_follower_ratio(self):
        ext = FeatureExtractor()
        bot_user = make_user(followers_count=5, following_count=5000)
        human_user = make_user(followers_count=1000, following_count=500)
        bot_vec = ext.extract_one(bot_user, [])
        human_vec = ext.extract_one(human_user, [])
        ratio_idx = FEATURE_NAMES.index("follower_ratio")
        assert bot_vec[ratio_idx] > human_vec[ratio_idx]

    def test_disinfo_keyword_density(self):
        ext = FeatureExtractor(keywords={"test": ["nato aggression"]})
        tweets_dis = [make_tweet("NATO aggression bad")] * 3
        tweets_clean = [make_tweet("I love coffee")] * 3
        dis_vec = ext.extract_one(make_user(), tweets_dis)
        clean_vec = ext.extract_one(make_user(), tweets_clean)
        dis_idx = FEATURE_NAMES.index("disinfo_keyword_density")
        assert dis_vec[dis_idx] > clean_vec[dis_idx]

    def test_heuristic_signals_passthrough(self):
        ext = FeatureExtractor()
        signals = {
            "account_age": 0.9,
            "username_pattern": 0.8,
            "profile_completeness": 0.0,
            "tweet_frequency": 0.5,
            "retweet_ratio": 0.7,
            "follower_ratio": 0.6,
            "content_similarity": 0.3,
            "posting_time_pattern": 0.2,
            "coordinated_behavior": 0.1,
            "disinformation_keywords": 0.4,
        }
        vec = ext.extract_one(make_user(), [], heuristic_signals=signals)
        sig_idx = FEATURE_NAMES.index("sig_account_age")
        assert vec[sig_idx] == pytest.approx(0.9)
        sig_idx2 = FEATURE_NAMES.index("sig_retweet_ratio")
        assert vec[sig_idx2] == pytest.approx(0.7)

    def test_extract_batch(self):
        ext = FeatureExtractor()
        records = [
            {
                "id": "t1", "text": "Hello", "author_id": "u1",
                "created_at": "2024-01-01T10:00:00+00:00",
                "referenced_tweets": [], "entities": {},
                "author": make_user(id="u1", username="alice"),
            },
            {
                "id": "t2", "text": "World", "author_id": "u2",
                "created_at": "2024-01-01T11:00:00+00:00",
                "referenced_tweets": [], "entities": {},
                "author": make_user(id="u2", username="bob"),
            },
        ]
        df = ext.extract_batch(records)
        assert df.shape == (2, len(FEATURE_NAMES))
        assert set(df.index) == {"u1", "u2"}
        assert list(df.columns) == FEATURE_NAMES

    def test_no_nan_in_output(self):
        ext = FeatureExtractor()
        user = make_user(account_age_days=None, followers_count=0)
        vec = ext.extract_one(user, [])
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isinf(vec))


class TestHourEntropy:
    def test_all_same_hour(self):
        from src.ml.feature_extractor import FeatureExtractor
        from datetime import datetime, timezone
        ext = FeatureExtractor()
        ts = [datetime(2024, 1, 1, 3, 0, tzinfo=timezone.utc)] * 10
        entropy = ext._hour_entropy(ts)
        assert entropy == pytest.approx(0.0)

    def test_uniform_24h(self):
        from src.ml.feature_extractor import FeatureExtractor
        from datetime import datetime, timezone
        ext = FeatureExtractor()
        ts = [datetime(2024, 1, 1, h, 0, tzinfo=timezone.utc) for h in range(24)]
        entropy = ext._hour_entropy(ts)
        assert entropy == pytest.approx(1.0, abs=0.01)

    def test_few_timestamps_returns_zero(self):
        ext = FeatureExtractor()
        assert ext._hour_entropy([]) == 0.0
