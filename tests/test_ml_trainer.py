"""Tests for the ML trainer and training data builder."""

import numpy as np
import pandas as pd
import pytest
from src.ml.feature_extractor import FEATURE_NAMES


def _make_training_df(n_bots: int = 40, n_humans: int = 40) -> pd.DataFrame:
    """Create a synthetic training DataFrame with distinguishable features."""
    rng = np.random.default_rng(42)

    bot_features = rng.uniform(0.5, 1.0, size=(n_bots, len(FEATURE_NAMES)))
    human_features = rng.uniform(0.0, 0.4, size=(n_humans, len(FEATURE_NAMES)))

    X = np.vstack([bot_features, human_features]).astype(np.float32)
    y = np.array([1] * n_bots + [0] * n_humans)
    conf = np.ones(len(y))

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y
    df["label_confidence"] = conf
    df["label_source"] = ["synthetic"] * len(y)
    return df


class TestBotClassifier:
    def test_train_returns_metrics(self):
        from src.ml.trainer import BotClassifier
        clf = BotClassifier()
        df = _make_training_df()
        metrics = clf.train(df, min_samples=10)
        assert "cv_roc_auc_mean" in metrics
        assert "cv_f1_mean" in metrics
        assert metrics["n_samples"] == 80

    def test_predict_proba_shape(self):
        from src.ml.trainer import BotClassifier
        clf = BotClassifier()
        df = _make_training_df()
        clf.train(df, min_samples=10)
        X = df[FEATURE_NAMES].values.astype(np.float32)
        probs = clf.predict_proba(X)
        assert probs.shape == (len(df),)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_predict_bots_score_higher(self):
        from src.ml.trainer import BotClassifier
        clf = BotClassifier()
        df = _make_training_df(n_bots=50, n_humans=50)
        clf.train(df, min_samples=10)
        X = df[FEATURE_NAMES].values.astype(np.float32)
        probs = clf.predict_proba(X)
        bot_probs = probs[:50]
        human_probs = probs[50:]
        assert bot_probs.mean() > human_probs.mean()

    def test_too_few_samples_raises(self):
        from src.ml.trainer import BotClassifier
        clf = BotClassifier()
        df = _make_training_df(n_bots=5, n_humans=5)
        with pytest.raises(ValueError, match="at least"):
            clf.train(df, min_samples=30)

    def test_save_and_load(self, tmp_path):
        from src.ml.trainer import BotClassifier
        clf = BotClassifier()
        df = _make_training_df()
        clf.train(df, min_samples=10)

        saved_path = clf.save(models_dir=tmp_path)
        assert saved_path.exists()
        assert (tmp_path / "latest.txt").exists()

        loaded = BotClassifier.load(models_dir=tmp_path)
        X = df[FEATURE_NAMES].values.astype(np.float32)
        probs = loaded.predict_proba(X)
        assert probs.shape == (len(df),)

    def test_is_trained_false_before_training(self, tmp_path):
        from src.ml.trainer import BotClassifier
        assert not BotClassifier.is_trained(tmp_path)

    def test_is_trained_true_after_save(self, tmp_path):
        from src.ml.trainer import BotClassifier
        clf = BotClassifier()
        df = _make_training_df()
        clf.train(df, min_samples=10)
        clf.save(models_dir=tmp_path)
        assert BotClassifier.is_trained(tmp_path)

    def test_predict_with_uncertainty(self):
        from src.ml.trainer import BotClassifier
        clf = BotClassifier()
        df = _make_training_df()
        clf.train(df, min_samples=10)
        X = df[FEATURE_NAMES].values[:5].astype(np.float32)
        result = clf.predict_with_uncertainty(X)
        assert "prob" in result
        assert "uncertainty" in result
        assert "lower" in result
        assert "upper" in result
        assert result["prob"].shape == (5,)


class TestTrainingDataBuilder:
    def test_from_synthetic_returns_labeled_df(self):
        from src.ml.training_data import TrainingDataBuilder
        builder = TrainingDataBuilder()
        df = builder.from_synthetic(n_accounts=30)
        assert "label" in df.columns
        assert "label_source" in df.columns
        assert set(df["label_source"].unique()) == {"synthetic"}
        assert df["label"].isin([0, 1]).all()
        # Should have both bots and humans
        assert 1 in df["label"].values
        assert 0 in df["label"].values

    def test_from_feedback_empty_when_no_file(self, tmp_path):
        from src.ml.training_data import TrainingDataBuilder
        builder = TrainingDataBuilder()
        df = builder.from_feedback(str(tmp_path / "nonexistent.json"))
        assert df.empty

    def test_merge_deduplicates(self):
        from src.ml.training_data import TrainingDataBuilder
        rng = np.random.default_rng(0)
        df1 = pd.DataFrame(
            rng.uniform(0, 1, (5, len(FEATURE_NAMES))).astype(np.float32),
            columns=FEATURE_NAMES,
            index=["u1", "u2", "u3", "u4", "u5"],
        )
        df1["label"] = 1
        df1["label_confidence"] = 1.0
        df1["label_source"] = "human"

        df2 = pd.DataFrame(
            rng.uniform(0, 1, (5, len(FEATURE_NAMES))).astype(np.float32),
            columns=FEATURE_NAMES,
            index=["u1", "u2", "u6", "u7", "u8"],  # u1, u2 overlap
        )
        df2["label"] = 0
        df2["label_confidence"] = 0.9
        df2["label_source"] = "synthetic"

        merged = TrainingDataBuilder.merge(df1, df2)
        assert len(merged) == 8  # 5 + 5 - 2 duplicates
        # Human labels should win for u1, u2
        assert merged.loc["u1", "label_source"] == "human"
        assert merged.loc["u2", "label_source"] == "human"

    def test_pseudo_labels_exclude_uncertain(self):
        from src.ml.training_data import TrainingDataBuilder, PSEUDO_BOT_THRESHOLD, PSEUDO_HUMAN_THRESHOLD
        from src.demo import generate_demo_data

        builder = TrainingDataBuilder()
        records = generate_demo_data(n_accounts=30)

        from src.detectors.bot_detector import BotDetector
        detector = BotDetector()
        heuristic_scores = detector.score_all(records)

        df = builder.from_heuristic_pseudolabels(records, heuristic_scores)
        # All included samples should be above/below thresholds
        for uid, row in df.iterrows():
            score_entry = next(
                (s for s in heuristic_scores if s.get("user_id") == uid), {}
            )
            score = score_entry.get("bot_score", 0.5)
            assert score >= PSEUDO_BOT_THRESHOLD or score <= PSEUDO_HUMAN_THRESHOLD
