"""
Active learning loop.

Strategy: Uncertainty sampling
-------------------------------
Accounts where the model is closest to 0.5 probability are the most
"uncertain" — and therefore the most valuable to label. Human review of
these accounts improves the model faster than reviewing easy cases.

The learner works in two modes:

  1. Pool-based (default)
     A batch of unlabelled accounts is scored; the N most uncertain are
     returned as a review queue.

  2. Stream-based
     As new tweets/accounts arrive, uncertainty scores are maintained and
     the review queue is updated dynamically.

The review queue is prioritised by:
  primary   : uncertainty (|prob - 0.5|, ascending)
  secondary : heuristic_score (descending) — prefer accounts the heuristic
              already thinks are bots, to maximise recall improvement
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReviewCandidate:
    user_id: str
    username: str
    ml_prob: float              # ML predicted P(bot)
    uncertainty: float          # |ml_prob - 0.5|; lower = more uncertain
    heuristic_score: float      # Raw heuristic bot score
    classification: str         # Current heuristic classification
    features: dict = field(default_factory=dict)
    signals: dict = field(default_factory=dict)
    top_tweets: list[str] = field(default_factory=list)

    @property
    def ml_pct(self) -> float:
        return round(self.ml_prob * 100, 1)

    @property
    def heuristic_pct(self) -> float:
        return round(self.heuristic_score * 100, 1)

    @property
    def disagreement(self) -> bool:
        """True when ML and heuristic strongly disagree."""
        h = self.heuristic_score
        m = self.ml_prob
        return abs(h - m) >= 0.25


class ActiveLearner:
    """Select the most informative accounts for human review."""

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("ml", {})
        self.uncertainty_window = cfg.get("uncertainty_window", 0.20)
        # Accounts with |prob - 0.5| < this are "uncertain"

    def select_review_queue(
        self,
        features_df: pd.DataFrame,
        ml_probs: np.ndarray,
        heuristic_scores: list[dict],
        records: list[dict],
        feedback_store=None,
        n: int = 20,
    ) -> list[ReviewCandidate]:
        """
        Return up to *n* ReviewCandidate objects sorted by uncertainty (most uncertain first).

        Parameters
        ----------
        features_df      : DataFrame with index = user_id
        ml_probs         : array of P(bot) aligned with features_df rows
        heuristic_scores : list of BotDetector.score_account() results
        records          : all tweet records (used to fetch sample tweets)
        feedback_store   : if provided, already-labelled accounts are excluded
        n                : max number of candidates to return
        """
        # Index lookups
        score_map = {
            s.get("user_id"): s for s in heuristic_scores
        }
        tweets_map = self._group_tweets(records)

        candidates: list[ReviewCandidate] = []

        for i, uid in enumerate(features_df.index):
            if feedback_store and (
                feedback_store.is_labelled(uid) or feedback_store.is_skipped(uid)
            ):
                continue

            prob = float(ml_probs[i])
            uncertainty = abs(prob - 0.5)

            hs = score_map.get(uid, {})
            heuristic_score = float(hs.get("bot_score", 0.5))
            classification = hs.get("classification", "unknown")

            # Feature dict for storage
            feat_dict = dict(zip(features_df.columns, features_df.iloc[i].tolist()))

            # Sample tweets for display
            sample_texts = [
                t.get("text", "")[:200]
                for t in tweets_map.get(uid, [])[:5]
                if t.get("text")
            ]

            candidates.append(ReviewCandidate(
                user_id=uid,
                username=(hs.get("author") or {}).get("username", uid),
                ml_prob=prob,
                uncertainty=uncertainty,
                heuristic_score=heuristic_score,
                classification=classification,
                features=feat_dict,
                signals=hs.get("signals", {}),
                top_tweets=sample_texts,
            ))

        # Sort by uncertainty ascending (most uncertain first),
        # then by heuristic score descending (prefer suspected bots)
        candidates.sort(key=lambda c: (c.uncertainty, -c.heuristic_score))

        # Prioritise candidates where ML and heuristic strongly disagree
        disagreements = [c for c in candidates if c.disagreement]
        agreements = [c for c in candidates if not c.disagreement]
        ordered = disagreements + agreements

        selected = ordered[:n]
        logger.info(
            "Review queue: %d candidates selected (of %d unlabelled), "
            "%d disagreements",
            len(selected),
            len(candidates),
            sum(1 for c in selected if c.disagreement),
        )
        return selected

    def select_for_stream(
        self,
        new_records: list[dict],
        ml_predictor,  # MLPredictor
        feedback_store,
        n: int = 10,
    ) -> list[ReviewCandidate]:
        """
        Stream mode: score newly arriving records and return uncertain ones.
        """
        results = ml_predictor.predict_batch(new_records)
        uncertain = [
            r for r in results
            if abs(r["ml_prob"] - 0.5) < self.uncertainty_window
            and not feedback_store.is_labelled(r["user_id"])
            and not feedback_store.is_skipped(r["user_id"])
        ]
        uncertain.sort(key=lambda r: abs(r["ml_prob"] - 0.5))

        return [
            ReviewCandidate(
                user_id=r["user_id"],
                username=r["username"],
                ml_prob=r["ml_prob"],
                uncertainty=abs(r["ml_prob"] - 0.5),
                heuristic_score=r["heuristic_score"],
                classification=r["classification"],
            )
            for r in uncertain[:n]
        ]

    @staticmethod
    def _group_tweets(records: list[dict]) -> dict[str, list[dict]]:
        from collections import defaultdict
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            uid = (r.get("author") or {}).get("id") or r.get("author_id", "")
            if uid:
                groups[uid].append(r)
        return groups
