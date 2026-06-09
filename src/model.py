# src/model.py - Completely fixed version (remove the artifact loading code at the bottom)
"""Model training and evaluation for return prediction."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
from sklearn.calibration import calibration_curve
from typing import Dict, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class ReturnPredictionModel:
    """Return prediction model with multiple algorithms."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'gradient_boosting': GradientBoostingClassifier(random_state=random_state, n_estimators=100),
            'random_forest': RandomForestClassifier(random_state=random_state, n_estimators=100)
        }
        self.trained_models = {}
        self.feature_names = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Train all models and return training metrics."""
        self.feature_names = X_train.columns.tolist()
        results = {}

        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model

            # Training metrics
            y_pred_proba = model.predict_proba(X_train)[:, 1]
            auc = roc_auc_score(y_train, y_pred_proba)
            results[name] = {'train_auc': auc}

        return results

    def predict_proba(self, X: pd.DataFrame, model_name: str = 'gradient_boosting') -> np.ndarray:
        """Predict return probabilities."""
        if model_name not in self.trained_models:
            raise ValueError(
                f"Model {model_name} not trained. Available: {list(self.trained_models.keys())}")

        model = self.trained_models[model_name]
        return model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.6) -> Dict:
        """Evaluate model performance."""
        metrics = {}

        for name, model in self.trained_models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= threshold).astype(int)

            # ROC-AUC
            auc = roc_auc_score(y_test, y_pred_proba)

            # Precision/Recall at threshold
            precision, recall, thresholds = precision_recall_curve(
                y_test, y_pred_proba)
            threshold_idx = np.argmin(
                np.abs(thresholds - threshold)) if len(thresholds) > 0 else -1

            metrics[name] = {
                'roc_auc': float(auc),
                'precision_at_threshold': float(precision[threshold_idx]) if threshold_idx >= 0 else 0,
                'recall_at_threshold': float(recall[threshold_idx]) if threshold_idx >= 0 else 0,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }

            # Calibration curve
            prob_true, prob_pred = calibration_curve(
                y_test, y_pred_proba, n_bins=10)
            metrics[name]['calibration'] = {
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist()
            }

        return metrics

    def get_feature_importance(self, model_name: str = 'gradient_boosting') -> pd.DataFrame:
        """Get feature importance for the specified model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.trained_models[model_name]

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df

    def explain_prediction(self, X_sample: pd.DataFrame, model_name: str = 'gradient_boosting') -> Dict:
        """Generate a simple explanation for a single prediction."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.trained_models[model_name]

        # Get prediction
        prediction = float(model.predict_proba(X_sample)[0, 1])

        # Simple feature contribution for tree-based models
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = X_sample.columns.tolist()

            # Get feature values for this sample
            feature_values = X_sample.iloc[0].values

            # Calculate contribution (simplified - value * importance)
            contributions = []
            for i in range(len(feature_names)):
                if i < len(feature_values) and i < len(importance):
                    contributions.append({
                        'feature': feature_names[i],
                        'contribution': float(feature_values[i] * importance[i]),
                        'importance': float(importance[i])
                    })

            contrib_df = pd.DataFrame(contributions)
            if len(contrib_df) > 0:
                contrib_df = contrib_df.sort_values(
                    'contribution', ascending=False)

                return {
                    'prediction_probability': prediction,
                    'top_risk_factors': contrib_df.head(5).to_dict('records'),
                    'top_reducing_factors': contrib_df.tail(5).to_dict('records')
                }

        return {'prediction_probability': prediction}
