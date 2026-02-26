"""
ML module – self-learning bot detection.

Pipeline:
  1. FeatureExtractor  – turns raw account+tweet records into a feature matrix
  2. TrainingDataBuilder – creates labeled datasets (synthetic + heuristic pseudo-labels)
  3. BotClassifier     – ensemble model (RF + GBT + LR) with calibrated probabilities
  4. ActiveLearner     – uncertainty sampling; surfaces edge cases for human review
  5. FeedbackStore     – persists human-confirmed labels; triggers auto-retraining
  6. MLPredictor       – unified predict() that uses ML when available, heuristic otherwise
"""
