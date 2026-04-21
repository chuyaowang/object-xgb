import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

try:
    import xgboost as xgb
except ImportError:
    xgb = None


class ObjectXGBoostClassifier:
    """
    XGBoost classifier for object-level classification using reduced features.
    Handles integer class labels (1, 2, 3...) for production.
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs : dict
            Hyperparameters for the XGBClassifier.
        """
        if xgb is None:
            raise ImportError(
                'XGBoost is not installed. Please run: pip install xgboost'
            )

        # Standard settings for multi-class research
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'eval_metric': 'mlogloss',
        }
        params.update(kwargs)
        self.model = xgb.XGBClassifier(**params)
        self.classes_ = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the XGBoost model on labeled data with class imbalance handling.

        Parameters
        ----------
        X : pd.DataFrame
            Reduced feature matrix.
        y : pd.Series
            Target labels (integers, non-zero).
        """
        mask = y > 0
        X_train = X[mask]
        y_train = y[mask]

        if len(X_train) == 0:
            raise ValueError('No labeled data available for training.')

        # XGBoost expects 0-indexed labels starting from 0
        unique_y = np.array(sorted(y_train.unique()))
        self.classes_ = unique_y
        label_map = {val: i for i, val in enumerate(unique_y)}
        y_train_mapped = y_train.map(label_map)

        # Handle class imbalance automatically
        print('[XGBoost] Calculating balanced sample weights...')
        weights = compute_sample_weight(class_weight='balanced', y=y_train)

        print(f'[XGBoost] Training on {len(X_train)} labeled samples...')
        self.model.fit(X_train, y_train_mapped, sample_weight=weights)
        print('[XGBoost] Training complete.')

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts integer classes for the input features.

        Returns
        -------
        y_pred : np.ndarray
            Predicted class labels (remapped to original integer IDs).
        """
        preds_zero_indexed = self.model.predict(X)
        # Map back to original integer IDs (1, 2, 3...)
        return np.array([self.classes_[i] for i in preds_zero_indexed])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns class probabilities."""
        return self.model.predict_proba(X)

    def predict_full_report(
        self, X_reduced: pd.DataFrame, y: pd.Series, original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generates a comprehensive report for all objects (labeled and unlabeled).

        Parameters
        ----------
        X_reduced : pd.DataFrame
            The feature matrix containing only selected features.
        y : pd.Series
            Target labels (true_label column), where 0 represents unlabeled.
        original_df : pd.DataFrame
            The original feature table containing 'slice_id' and 'label'.

        Returns
        -------
        report_df : pd.DataFrame
            DataFrame containing metadata, true labels, and predicted labels (all integers).
        """
        print(
            f'[XGBoost] Generating full report for {len(X_reduced)} objects...'
        )

        # 1. Predict for all objects
        all_preds = self.predict(X_reduced)

        # 2. Construct the report
        report_df = pd.DataFrame(
            {
                'slice_id': original_df['slice_id'],
                'label': original_df['label'],
                'true_label': y.fillna(0).values.astype(int),
                'predicted_label': all_preds.astype(int),
            }
        )

        return report_df
