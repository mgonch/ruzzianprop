"""
Feedback store – persists human-confirmed labels between sessions.

File format: data/feedback_labels.json
{
  "version": 1,
  "labels": [
    {
      "user_id":  "123456789",
      "username": "someuser",
      "label":    1,           // 1 = bot, 0 = human
      "label_str": "bot",
      "reviewer_note": "...",
      "reviewed_at": "2024-01-01T12:00:00Z",
      "features": {<FEATURE_NAMES: values>},
      "heuristic_score": 0.82
    },
    ...
  ],
  "retrain_trigger": 10,      // Retrain after this many new labels
  "new_labels_since_train": 0
}

Auto-retraining
---------------
When new_labels_since_train >= retrain_trigger, the store returns
needs_retrain=True from add_label(). The caller (CLI) then invokes
the trainer.
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PATH = "data/feedback_labels.json"
DEFAULT_RETRAIN_TRIGGER = 10


class FeedbackStore:
    """Persist and retrieve human-confirmed bot/human labels."""

    def __init__(
        self,
        path: str = DEFAULT_PATH,
        retrain_trigger: int = DEFAULT_RETRAIN_TRIGGER,
    ):
        self.path = Path(path)
        self.retrain_trigger = retrain_trigger
        self._data: dict = self._load()

    # ------------------------------------------------------------------

    def add_label(
        self,
        user_id: str,
        username: str,
        label: int,
        features: dict,
        heuristic_score: float = 0.0,
        reviewer_note: str = "",
    ) -> bool:
        """
        Record a human-confirmed label.

        Returns True if the model should be retrained now.
        """
        entry = {
            "user_id": user_id,
            "username": username,
            "label": label,
            "label_str": "bot" if label == 1 else "human",
            "reviewer_note": reviewer_note,
            "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "features": features,
            "heuristic_score": round(heuristic_score, 4),
        }

        # Update or insert (indexed by user_id)
        existing_ids = {e["user_id"] for e in self._data["labels"]}
        if user_id in existing_ids:
            self._data["labels"] = [
                e if e["user_id"] != user_id else entry
                for e in self._data["labels"]
            ]
            logger.info("Updated label for @%s → %s", username, entry["label_str"])
        else:
            self._data["labels"].append(entry)
            self._data["new_labels_since_train"] += 1
            logger.info("Added label for @%s → %s", username, entry["label_str"])

        self._save()

        needs_retrain = self._data["new_labels_since_train"] >= self._data["retrain_trigger"]
        if needs_retrain:
            logger.info(
                "Reached retrain trigger (%d new labels). Model should be retrained.",
                self._data["new_labels_since_train"],
            )
        return needs_retrain

    def mark_retrained(self) -> None:
        """Call after a successful retraining to reset the counter."""
        self._data["new_labels_since_train"] = 0
        self._save()

    def get_all(self) -> list[dict]:
        return list(self._data["labels"])

    def count(self) -> int:
        return len(self._data["labels"])

    def count_by_label(self) -> dict[str, int]:
        labels = [e["label"] for e in self._data["labels"]]
        return {"bot": sum(1 for l in labels if l == 1),
                "human": sum(1 for l in labels if l == 0)}

    def new_since_train(self) -> int:
        return self._data["new_labels_since_train"]

    def skip(self, user_id: str) -> None:
        """Mark an account as skipped (won't surface again in review queue)."""
        self._data.setdefault("skipped", set())
        if isinstance(self._data["skipped"], list):
            self._data["skipped"] = set(self._data["skipped"])
        self._data["skipped"].add(user_id)
        self._data_to_disk()

    def is_skipped(self, user_id: str) -> bool:
        skipped = self._data.get("skipped", [])
        return user_id in skipped

    def is_labelled(self, user_id: str) -> bool:
        return any(e["user_id"] == user_id for e in self._data["labels"])

    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                data.setdefault("retrain_trigger", self.retrain_trigger)
                data.setdefault("new_labels_since_train", 0)
                data.setdefault("labels", [])
                return data
            except Exception as e:
                logger.warning("Could not load feedback file: %s", e)

        return {
            "version": 1,
            "labels": [],
            "retrain_trigger": self.retrain_trigger,
            "new_labels_since_train": 0,
            "skipped": [],
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Convert set → list for JSON serialisation
        data_copy = dict(self._data)
        if isinstance(data_copy.get("skipped"), set):
            data_copy["skipped"] = list(data_copy["skipped"])
        with open(self.path, "w") as f:
            json.dump(data_copy, f, indent=2)

    def _data_to_disk(self) -> None:
        self._save()
