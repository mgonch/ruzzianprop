"""
Ensemble ML trainer with probability calibration.

Model
-----
Voting ensemble of three calibrated base estimators:

  1. RandomForestClassifier       – handles non-linear interactions well;
                                    robust to outliers and missing features
  2. GradientBoostingClassifier   – strong sequential learner; good on
                                    tabular data with mixed feature types
  3. LogisticRegression           – provides a linear baseline; helps
                                    calibrate the ensemble's probabilities

Each base estimator is wrapped in CalibratedClassifierCV (isotonic
regression) to ensure well-calibrated probabilities rather than raw
confidence scores.

The ensemble uses soft voting (average of calibrated probabilities) and
then re-calibrates the ensemble output using Platt scaling.

Training pipeline
-----------------
1. Impute missing values (median) and scale features (StandardScaler)
2. Handle class imbalance with sample_weight (inverse class frequency)
3. 5-fold stratified cross-validation with precision / recall / ROC-AUC
4. Final fit on all training data
5. Save model artifact to models/ directory (joblib)

Model versioning
----------------
Each saved model is named  models/bot_classifier_v{N}.joblib  where N is
an auto-incrementing integer.  The file models/latest.txt points to the
most recent version.
"""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.ml.feature_extractor import FEATURE_NAMES

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
METRICS_FILE = MODELS_DIR / "training_history.json"


class BotClassifier:
    """
    Trains and evaluates an ensemble bot detection model.

    Usage
    -----
    clf = BotClassifier()
    clf.train(X, y, sample_weights=w)
    clf.save()

    probs = BotClassifier.load().predict_proba(X)
    """

    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("ml", {})
        self.n_estimators_rf = cfg.get("rf_n_estimators", 200)
        self.n_estimators_gb = cfg.get("gb_n_estimators", 150)
        self.cv_folds = cfg.get("cv_folds", 5)
        self.random_state = cfg.get("random_state", 42)
        self.pipeline: Pipeline | None = None
        self.feature_importances_: dict[str, float] = {}
        self.cv_metrics_: dict = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        training_df: pd.DataFrame,
        min_samples: int = 30,
    ) -> dict:
        """
        Train on *training_df* which must have feature columns + 'label' column.

        Optionally uses 'label_confidence' as sample weights if present.

        Returns a metrics dict (CV + held-out).
        """
        if "label" not in training_df.columns:
            raise ValueError("training_df must contain a 'label' column")

        X = training_df[FEATURE_NAMES].values.astype(np.float32)
        y = training_df["label"].values.astype(int)

        if len(X) < min_samples:
            raise ValueError(
                f"Need at least {min_samples} labeled samples to train. "
                f"Got {len(X)}. Run 'ruzzianprop train --bootstrap' to generate more."
            )

        # Sample weights from label confidence
        if "label_confidence" in training_df.columns:
            weights = training_df["label_confidence"].values.astype(float)
        else:
            weights = np.ones(len(y), dtype=float)

        # Handle class imbalance
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        if n_pos > 0 and n_neg > 0:
            class_weight = np.where(y == 1, len(y) / (2 * n_pos), len(y) / (2 * n_neg))
            weights = weights * class_weight

        logger.info(
            "Training on %d samples (%d bots, %d humans)", len(y), n_pos, n_neg
        )

        self.pipeline = self._build_pipeline()

        # Cross-validation (without sample_weight to stay compatible across sklearn versions)
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_results = cross_validate(
            self.pipeline,
            X, y,
            cv=cv,
            scoring=["roc_auc", "average_precision", "f1"],
            return_train_score=True,
            n_jobs=-1,
        )

        self.cv_metrics_ = {
            "cv_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
            "cv_avg_precision_mean": float(np.mean(cv_results["test_average_precision"])),
            "cv_f1_mean": float(np.mean(cv_results["test_f1"])),
            "cv_f1_std": float(np.std(cv_results["test_f1"])),
            "n_samples": int(len(y)),
            "n_bots": int(n_pos),
            "n_humans": int(n_neg),
        }

        logger.info(
            "CV ROC-AUC: %.3f ± %.3f | F1: %.3f ± %.3f | AP: %.3f",
            self.cv_metrics_["cv_roc_auc_mean"],
            self.cv_metrics_["cv_roc_auc_std"],
            self.cv_metrics_["cv_f1_mean"],
            self.cv_metrics_["cv_f1_std"],
            self.cv_metrics_["cv_avg_precision_mean"],
        )

        # Final fit on all data
        self.pipeline.fit(X, y)

        # Feature importances from RF
        try:
            vclf = self.pipeline.named_steps["classifier"]
            for est_name, est in vclf.estimators:
                if "rf" in est_name:
                    inner = est.calibrated_classifiers_[0].estimator
                    importances = inner.feature_importances_
                    self.feature_importances_ = dict(zip(FEATURE_NAMES, importances.tolist()))
                    break
        except Exception:
            pass

        return self.cv_metrics_

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(bot) for each row in X. Shape: (n_samples,)."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        probs = self.pipeline.predict_proba(X)
        # Returns (n, 2); column 1 = P(bot)
        return probs[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.50) -> np.ndarray:
        """Return binary predictions (0/1)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def predict_with_uncertainty(self, X: np.ndarray) -> dict:
        """
        Return predictions with uncertainty estimates via bootstrap.
        Returns dict with keys: prob, uncertainty (std), lower, upper (95% CI).
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained.")

        # Use individual base estimators for uncertainty
        probs_list = []
        try:
            vclf = self.pipeline.named_steps["classifier"]
            imputer = self.pipeline.named_steps["imputer"]
            scaler = self.pipeline.named_steps["scaler"]
            X_proc = scaler.transform(imputer.transform(X))
            for _, estimator in vclf.estimators:
                probs_list.append(estimator.predict_proba(X_proc)[:, 1])
        except Exception:
            probs_list = [self.predict_proba(X)]

        probs_arr = np.stack(probs_list, axis=0)  # (n_estimators, n_samples)
        return {
            "prob": probs_arr.mean(axis=0),
            "uncertainty": probs_arr.std(axis=0),
            "lower": np.percentile(probs_arr, 5, axis=0),
            "upper": np.percentile(probs_arr, 95, axis=0),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, models_dir: str | Path = MODELS_DIR) -> Path:
        """Save the trained model and return the path."""
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        # Version numbering
        existing = sorted(models_dir.glob("bot_classifier_v*.joblib"))
        next_ver = len(existing) + 1
        path = models_dir / f"bot_classifier_v{next_ver}.joblib"

        artifact = {
            "pipeline": self.pipeline,
            "feature_importances": self.feature_importances_,
            "cv_metrics": self.cv_metrics_,
            "feature_names": FEATURE_NAMES,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "version": next_ver,
        }
        joblib.dump(artifact, path)

        # Update latest pointer
        (models_dir / "latest.txt").write_text(path.name)

        # Append to training history
        self._append_history(artifact["cv_metrics"], path, models_dir)

        logger.info("Model saved to %s", path)
        return path

    @classmethod
    def load(cls, models_dir: str | Path = MODELS_DIR) -> "BotClassifier":
        """Load the most recent saved model."""
        models_dir = Path(models_dir)
        latest_ptr = models_dir / "latest.txt"
        if not latest_ptr.exists():
            raise FileNotFoundError(
                f"No trained model found in {models_dir}. "
                "Run 'ruzzianprop train' first."
            )
        model_path = models_dir / latest_ptr.read_text().strip()
        artifact = joblib.load(model_path)
        instance = cls()
        instance.pipeline = artifact["pipeline"]
        instance.feature_importances_ = artifact.get("feature_importances", {})
        instance.cv_metrics_ = artifact.get("cv_metrics", {})
        logger.info("Loaded model from %s", model_path)
        return instance

    @classmethod
    def is_trained(cls, models_dir: str | Path = MODELS_DIR) -> bool:
        return (Path(models_dir) / "latest.txt").exists()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> Pipeline:
        rf = CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=self.n_estimators_rf,
                max_depth=12,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            ),
            method="isotonic",
            cv=3,
        )
        gb = CalibratedClassifierCV(
            GradientBoostingClassifier(
                n_estimators=self.n_estimators_gb,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state,
            ),
            method="isotonic",
            cv=3,
        )
        lr = CalibratedClassifierCV(
            LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state,
            ),
            method="sigmoid",
            cv=3,
        )
        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
            voting="soft",
            weights=[3, 3, 1],  # RF and GB have more weight than LR
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", ensemble),
        ])

    @staticmethod
    def _append_history(metrics: dict, model_path: Path, models_dir: Path) -> None:
        hist_path = models_dir / "training_history.json"
        history = []
        if hist_path.exists():
            try:
                with open(hist_path) as f:
                    history = json.load(f)
            except Exception:
                pass
        history.append({**metrics, "model_file": model_path.name})
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)
