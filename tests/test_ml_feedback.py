"""Tests for FeedbackStore and ActiveLearner."""

import json
import pytest
from src.ml.feedback_store import FeedbackStore
from src.ml.active_learner import ActiveLearner, ReviewCandidate


class TestFeedbackStore:
    def test_add_label_persists(self, tmp_path):
        store = FeedbackStore(path=str(tmp_path / "labels.json"), retrain_trigger=5)
        needs_retrain = store.add_label(
            user_id="u1", username="alice", label=1,
            features={"account_age_days": 10.0}, heuristic_score=0.85,
        )
        assert not needs_retrain
        assert store.count() == 1

    def test_retrain_trigger(self, tmp_path):
        store = FeedbackStore(path=str(tmp_path / "labels.json"), retrain_trigger=3)
        for i in range(2):
            needs = store.add_label(
                user_id=f"u{i}", username=f"user{i}", label=1,
                features={}, heuristic_score=0.8,
            )
            assert not needs
        # 3rd label should trigger
        needs = store.add_label(
            user_id="u99", username="final", label=0,
            features={}, heuristic_score=0.1,
        )
        assert needs

    def test_update_existing_label(self, tmp_path):
        store = FeedbackStore(path=str(tmp_path / "labels.json"), retrain_trigger=10)
        store.add_label("u1", "alice", 1, {}, 0.9)
        store.add_label("u1", "alice", 0, {}, 0.1)  # Update
        entries = [e for e in store.get_all() if e["user_id"] == "u1"]
        assert len(entries) == 1
        assert entries[0]["label"] == 0

    def test_is_labelled(self, tmp_path):
        store = FeedbackStore(path=str(tmp_path / "labels.json"), retrain_trigger=10)
        store.add_label("u1", "alice", 1, {}, 0.9)
        assert store.is_labelled("u1")
        assert not store.is_labelled("u2")

    def test_skip(self, tmp_path):
        store = FeedbackStore(path=str(tmp_path / "labels.json"), retrain_trigger=10)
        store.skip("u1")
        assert store.is_skipped("u1")
        assert not store.is_skipped("u2")

    def test_count_by_label(self, tmp_path):
        store = FeedbackStore(path=str(tmp_path / "labels.json"), retrain_trigger=10)
        store.add_label("u1", "alice", 1, {}, 0.9)
        store.add_label("u2", "bob", 0, {}, 0.1)
        store.add_label("u3", "carol", 1, {}, 0.8)
        counts = store.count_by_label()
        assert counts["bot"] == 2
        assert counts["human"] == 1

    def test_mark_retrained_resets_counter(self, tmp_path):
        store = FeedbackStore(path=str(tmp_path / "labels.json"), retrain_trigger=5)
        for i in range(3):
            store.add_label(f"u{i}", f"u{i}", 1, {}, 0.9)
        assert store.new_since_train() == 3
        store.mark_retrained()
        assert store.new_since_train() == 0

    def test_persistence_across_instances(self, tmp_path):
        path = str(tmp_path / "labels.json")
        store1 = FeedbackStore(path=path, retrain_trigger=10)
        store1.add_label("u1", "alice", 1, {"feat": 0.5}, 0.9)

        store2 = FeedbackStore(path=path, retrain_trigger=10)
        assert store2.count() == 1
        assert store2.is_labelled("u1")


class TestActiveLearner:
    def test_select_review_queue_most_uncertain_first(self):
        import numpy as np
        import pandas as pd
        from src.ml.feature_extractor import FEATURE_NAMES

        learner = ActiveLearner()

        # Create fake feature df with 5 accounts
        uids = ["u1", "u2", "u3", "u4", "u5"]
        df = pd.DataFrame(
            np.zeros((5, len(FEATURE_NAMES)), dtype=np.float32),
            columns=FEATURE_NAMES,
            index=uids,
        )

        # Use ML probs close to heuristic (0.5) to avoid disagreement prioritisation.
        # u3 (prob=0.52) is closest to 0.5 → most uncertain → should rank first.
        ml_probs = np.array([0.80, 0.70, 0.52, 0.35, 0.10])
        # Heuristic scores close to ML probs so |h - ml| < 0.25 for all
        heuristic_scores = [
            {"user_id": uid, "bot_score": float(p), "bot_score_pct": float(p * 100),
             "classification": "suspected", "signals": {}, "author": {"username": uid}}
            for uid, p in zip(uids, ml_probs)
        ]

        queue = learner.select_review_queue(
            features_df=df,
            ml_probs=ml_probs,
            heuristic_scores=heuristic_scores,
            records=[],
            n=5,
        )
        assert len(queue) == 5
        # u3 (prob=0.52, uncertainty=0.02) should rank first among non-disagreements
        assert queue[0].user_id == "u3"

    def test_already_labelled_excluded(self, tmp_path):
        import numpy as np
        import pandas as pd
        from src.ml.feature_extractor import FEATURE_NAMES

        store = FeedbackStore(path=str(tmp_path / "fb.json"), retrain_trigger=10)
        store.add_label("u1", "alice", 1, {}, 0.9)

        learner = ActiveLearner()
        uids = ["u1", "u2"]
        df = pd.DataFrame(
            np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32),
            columns=FEATURE_NAMES,
            index=uids,
        )
        ml_probs = np.array([0.5, 0.5])
        heuristic_scores = [
            {"user_id": uid, "bot_score": 0.5, "classification": "suspected",
             "signals": {}, "author": {"username": uid}}
            for uid in uids
        ]

        queue = learner.select_review_queue(
            df, ml_probs, heuristic_scores, [],
            feedback_store=store, n=10,
        )
        # u1 is already labelled and should be excluded
        assert all(c.user_id != "u1" for c in queue)
        assert len(queue) == 1

    def test_disagreement_flagged(self):
        candidate = ReviewCandidate(
            user_id="u1", username="bot", ml_prob=0.2,
            uncertainty=0.3, heuristic_score=0.9,
            classification="bot",
        )
        assert candidate.disagreement  # |0.9 - 0.2| = 0.7 >= 0.25

    def test_agreement_not_flagged(self):
        candidate = ReviewCandidate(
            user_id="u1", username="bot", ml_prob=0.85,
            uncertainty=0.15, heuristic_score=0.9,
            classification="bot",
        )
        assert not candidate.disagreement  # |0.9 - 0.85| = 0.05 < 0.25
