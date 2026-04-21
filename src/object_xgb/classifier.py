import numpy as np
import pandas as pd

from .augmentation import FeatureAugmentor
from .feature_selection import PairwisePLSFeatureSelector
from .xgboost_classifier import ObjectXGBoostClassifier as XGBWrapper


class ObjectClassifier:
    """
    Production classifier for the object-xgb plugin.
    Integrates PLS-DA feature selection and XGBoost using integer labels.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        use_augmentation: bool = False,
        balance_classes: bool = False,
        augmentation_params: dict = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        threshold : float
            VIP threshold for PLS-DA feature selection.
        use_augmentation : bool
            Whether to augment labeled data during training.
        balance_classes : bool
            Whether to balance minority classes using SMOTE-style interpolation.
        augmentation_params : dict
            Hyperparameters for the FeatureAugmentor.
        **kwargs : dict
            Hyperparameters for the XGBoost model.
        """
        self.selector = PairwisePLSFeatureSelector(threshold=threshold)
        self.model = XGBWrapper(**kwargs)
        self.selected_features = []
        self.use_augmentation = use_augmentation
        self.balance_classes = balance_classes
        self.augmentor = (
            FeatureAugmentor(**(augmentation_params or {}))
            if (use_augmentation or balance_classes)
            else None
        )

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Executes the two-stage training pipeline.
        1. Identifies discriminating features via pairwise PLS-DA.
        2. Trains an XGBoost model on the (optionally augmented) selected subset.
        """
        print('[Object XGB] Starting feature selection (Pairwise PLS-DA)...')
        X_red = self.selector.fit_transform(X, y)
        self.selected_features = self.selector.selected_features

        # Filter for labeled data
        mask = y > 0
        X_train = X_red[mask]
        y_train = y[mask]

        if self.augmentor and (self.use_augmentation or self.balance_classes):
            print(
                f'[Object XGB] Augmenting/Balancing {len(X_train)} labeled samples...'
            )
            X_train, y_train = self.augmentor.augment(
                X_train, y_train, balance=self.balance_classes
            )

        print(
            f'[Object XGB] Training XGBoost on {len(X_train)} samples ({len(self.selected_features)} features)...'
        )
        self.model.train(X_train, y_train)
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
