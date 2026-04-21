import numpy as np
import pandas as pd

from .feature_selection import PairwisePLSFeatureSelector
from .xgboost_classifier import ObjectXGBoostClassifier as XGBWrapper


class ObjectClassifier:
    """
    Production classifier for the object-xgb plugin.
    Integrates PLS-DA feature selection and XGBoost using integer labels.
    """

    def __init__(self, threshold: float = 1.0, **kwargs):
        """
        Parameters
        ----------
        threshold : float
            VIP threshold for PLS-DA feature selection.
        **kwargs : dict
            Hyperparameters for the XGBoost model.
        """
        self.selector = PairwisePLSFeatureSelector(threshold=threshold)
        self.model = XGBWrapper(**kwargs)
        self.selected_features = []

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Executes the two-stage training pipeline.
        1. Identifies discriminating features via pairwise PLS-DA.
        2. Trains an XGBoost model on the selected subset.
        """
        print('[Object XGB] Starting feature selection (Pairwise PLS-DA)...')
        X_red = self.selector.fit_transform(X, y)
        self.selected_features = self.selector.selected_features

        print(
            f'[Object XGB] Training XGBoost on {len(self.selected_features)} features...'
        )
        self.model.train(X_red, y)
        print('[Object XGB] Pipeline training complete.')

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts integer classes using only the selected features."""
        if not self.selected_features:
            return self.model.predict(X)
        return self.model.predict(X[self.selected_features])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts probabilities using only the selected features."""
        if not self.selected_features:
            return self.model.predict_proba(X)
        return self.model.predict_proba(X[self.selected_features])

    def get_report(
        self, X: pd.DataFrame, y: pd.Series, original_df: pd.DataFrame
    ):
        """Generates a complete prediction report table with integer labels."""
        X_red = X[self.selected_features] if self.selected_features else X
        return self.model.predict_full_report(X_red, y, original_df)
