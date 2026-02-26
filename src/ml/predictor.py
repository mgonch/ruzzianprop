"""
ML predictor – unified predict() interface.

Decision logic
--------------
1. If a trained ML model exists → use ML probability, enriched with
   uncertainty bounds from the ensemble's individual estimators.
2. If no ML model exists → fall back to the heuristic BotDetector score.
3. The final output always includes BOTH scores so analysts can see where
   they agree / disagree.

The predictor also maintains a self-improvement feedback loop:
  - After every prediction batch, accounts near the decision boundary
    (40%–60% ML probability) are queued for human review via the ActiveLearner.
  - When the FeedbackStore accumulates enough new labels it signals retraining.
  - Retraining can be triggered automatically (auto_retrain=True) or manually.
"""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    Unified prediction interface that combines ML and heuristic scoring.

    Parameters
    ----------
    config         : full YAML config dict
    models_dir     : directory where trained model artifacts live
    feedback_path  : path to the FeedbackStore JSON file
    auto_retrain   : if True, retrain when FeedbackStore signals it
    """

    def __init__(
        self,
        config: dict | None = None,
        models_dir: str = "models",
        feedback_path: str = "data/feedback_labels.json",
        auto_retrain: bool = True,
    ):
        self.config = config or {}
        self.models_dir = Path(models_dir)
        self.feedback_path = feedback_path
        self.auto_retrain = auto_retrain
        self._ml_model = None
        self._extractor = None
        self._feedback_store = None

    # ------------------------------------------------------------------
    # Lazy-loaded components
    # ------------------------------------------------------------------

    @property
    def ml_model(self):
        if self._ml_model is None:
            from src.ml.trainer import BotClassifier
            if BotClassifier.is_trained(self.models_dir):
                try:
                    self._ml_model = BotClassifier.load(self.models_dir)
                    logger.info("ML model loaded from %s", self.models_dir)
                except Exception as e:
                    logger.warning("Failed to load ML model: %s", e)
        return self._ml_model

    @property
    def extractor(self):
        if self._extractor is None:
            from src.ml.feature_extractor import FeatureExtractor
            dis_cfg = self.config.get("disinformation", {})
            self._extractor = FeatureExtractor(
                keywords=dis_cfg.get("keywords", {}),
                state_media=set(dis_cfg.get("state_media_domains", [])),
            )
        return self._extractor

    @property
    def feedback_store(self):
        if self._feedback_store is None:
            from src.ml.feedback_store import FeedbackStore
            self._feedback_store = FeedbackStore(path=self.feedback_path)
        return self._feedback_store

    # ------------------------------------------------------------------
    # Main prediction interface
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        records: list[dict],
        heuristic_scores: list[dict] | None = None,
        graph=None,
    ) -> list[dict]:
        """
        Score all unique authors in *records*.

        Returns a list of result dicts, one per account, sorted by
        combined_score descending. Each dict has:
          user_id, username, classification,
          ml_prob, ml_pct, ml_uncertainty, ml_lower, ml_upper,
          heuristic_score, heuristic_pct,
          combined_score, combined_pct,
          source (ml | heuristic),
          signals (from heuristic),
          author (raw user dict)
        """
        from src.detectors.bot_detector import BotDetector

        # Run heuristic if not pre-computed
        if heuristic_scores is None:
            detector = BotDetector(
                config=self.config,
                keywords=self.config.get("disinformation", {}).get("keywords", {}),
            )
            heuristic_scores = detector.score_all(records)

        heuristic_map = {
            s.get("user_id"): s for s in heuristic_scores
        }

        # Extract features
        features_df = self.extractor.extract_batch(
            records,
            heuristic_scores=heuristic_scores,
            graph=graph,
        )

        results: list[dict] = []
        ml_available = self.ml_model is not None

        if ml_available and not features_df.empty:
            X = features_df.values.astype(np.float32)
            try:
                uncertainty_result = self.ml_model.predict_with_uncertainty(X)
                ml_probs = uncertainty_result["prob"]
                ml_uncertainty = uncertainty_result["uncertainty"]
                ml_lower = uncertainty_result["lower"]
                ml_upper = uncertainty_result["upper"]
            except Exception as e:
                logger.warning("ML prediction failed: %s – falling back to heuristic", e)
                ml_available = False
                ml_probs = np.zeros(len(features_df))
                ml_uncertainty = ml_lower = ml_upper = ml_probs
        else:
            ml_probs = np.zeros(len(features_df))
            ml_uncertainty = ml_lower = ml_upper = ml_probs

        # Group tweets for display
        tweets_by_author: dict[str, list] = defaultdict(list)
        for r in records:
            uid = (r.get("author") or {}).get("id") or r.get("author_id", "")
            if uid:
                tweets_by_author[uid].append(r)

        for i, uid in enumerate(features_df.index):
            hs = heuristic_map.get(uid, {})
            h_score = float(hs.get("bot_score", 0.0))
            h_pct = float(hs.get("bot_score_pct", 0.0))

            if ml_available:
                m_prob = float(ml_probs[i])
                m_unc = float(ml_uncertainty[i])
                m_lower = float(ml_lower[i])
                m_upper = float(ml_upper[i])
                # Weighted combination: 60% ML, 40% heuristic
                combined = 0.60 * m_prob + 0.40 * h_score
                source = "ml"
            else:
                m_prob = h_score
                m_unc = 0.0
                m_lower = h_score
                m_upper = h_score
                combined = h_score
                source = "heuristic"

            combined_pct = round(combined * 100, 1)

            if combined >= 0.60:
                classification = "bot"
            elif combined >= 0.40:
                classification = "suspected"
            else:
                classification = "human"

            results.append({
                "user_id": uid,
                "username": (hs.get("author") or {}).get("username", uid),
                "classification": classification,
                "ml_prob": round(m_prob, 4),
                "ml_pct": round(m_prob * 100, 1),
                "ml_uncertainty": round(m_unc, 4),
                "ml_lower": round(m_lower * 100, 1),
                "ml_upper": round(m_upper * 100, 1),
                "heuristic_score": round(h_score, 4),
                "heuristic_pct": round(h_pct, 1),
                "combined_score": round(combined, 4),
                "combined_pct": combined_pct,
                "source": source,
                "signals": hs.get("signals", {}),
                "author": hs.get("author", {}),
                "features": dict(zip(features_df.columns, features_df.iloc[i].tolist())),
            })

        results.sort(key=lambda r: r["combined_score"], reverse=True)

        # Check if we should auto-retrain
        if self.auto_retrain and self.feedback_store.new_since_train() >= self.feedback_store.retrain_trigger:
            logger.info("Auto-retraining triggered by FeedbackStore...")
            self._auto_retrain(records, heuristic_scores)

        return results

    def predict_one(self, user: dict, tweets: list[dict]) -> dict:
        """Score a single account."""
        record = {
            "id": "single",
            "text": (tweets[0]["text"] if tweets else ""),
            "author_id": user.get("id", ""),
            "author": user,
            "referenced_tweets": [],
            "entities": {},
        }
        results = self.predict_batch([record] + tweets)
        return results[0] if results else {}

    # ------------------------------------------------------------------
    # Active learning integration
    # ------------------------------------------------------------------

    def get_review_queue(
        self,
        records: list[dict],
        heuristic_scores: list[dict],
        n: int = 20,
    ) -> list:
        """Return the top *n* uncertain accounts for human review."""
        from src.ml.active_learner import ActiveLearner
        from src.ml.feature_extractor import FEATURE_NAMES

        if self.ml_model is None:
            logger.warning("No ML model loaded. Train first with 'ruzzianprop train'.")
            return []

        features_df = self.extractor.extract_batch(records, heuristic_scores=heuristic_scores)
        if features_df.empty:
            return []

        X = features_df.values.astype(np.float32)
        ml_probs = self.ml_model.predict_proba(X)

        learner = ActiveLearner(self.config)
        return learner.select_review_queue(
            features_df=features_df,
            ml_probs=ml_probs,
            heuristic_scores=heuristic_scores,
            records=records,
            feedback_store=self.feedback_store,
            n=n,
        )

    # ------------------------------------------------------------------
    # Auto-retraining
    # ------------------------------------------------------------------

    def _auto_retrain(self, records: list[dict], heuristic_scores: list[dict]) -> None:
        try:
            from src.ml.training_data import TrainingDataBuilder
            from src.ml.trainer import BotClassifier

            builder = TrainingDataBuilder(self.config)
            feedback_df = builder.from_feedback(self.feedback_path)
            synthetic_df = builder.from_synthetic(n_accounts=150)
            pseudo_df = builder.from_heuristic_pseudolabels(records, heuristic_scores)
            training_df = TrainingDataBuilder.merge(feedback_df, synthetic_df, pseudo_df)

            clf = BotClassifier(self.config)
            clf.train(training_df)
            clf.save(self.models_dir)

            # Reload
            self._ml_model = clf
            self.feedback_store.mark_retrained()
            logger.info("Auto-retraining complete.")
        except Exception as e:
            logger.error("Auto-retraining failed: %s", e)
