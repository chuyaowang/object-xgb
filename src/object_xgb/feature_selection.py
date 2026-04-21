import pandas as pd

from .pls_analysis import analyze_all_pairs


class PairwisePLSFeatureSelector:
    """
    Feature selector that uses pairwise PLS-DA VIP scores to identify
    the most discriminative features.
    """

    def __init__(self, threshold: float = 1.0):
        """
        Parameters
        ----------
        threshold : float
            VIP score threshold. Features with a VIP score > threshold in
            at least one pairwise comparison will be selected.
        """
        self.threshold = threshold
        self.selected_features = []
        self.pair_vips = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Calculates pairwise VIP scores and identifies significant features.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (labeled data only).
        y : pd.Series
            Target labels (non-zero).
        """
        # Ensure we only use labeled data for selection
        mask = y > 0
        if not mask.any():
            raise ValueError('No labeled data found for feature selection.')

        X_lab = X[mask]
        y_lab = y[mask]

        # Get pairwise VIPs using our existing analysis logic
        self.pair_vips = analyze_all_pairs(X_lab, y_lab, X.columns.tolist())

        # Safety Logic: Cap the threshold at the maximum calculated VIP score
        # This prevents selecting zero features if the user sets the slider too high.
        max_vip = self.pair_vips.values.max()
        effective_threshold = self.threshold
        if effective_threshold >= max_vip:
            effective_threshold = (
                max_vip * 0.99
            )  # Slightly below max to pick at least one
            print(
                f'[Feature Selection] User threshold {self.threshold:.2f} exceeds max VIP {max_vip:.2f}. '
                f'Capping at {effective_threshold:.2f}'
            )

        # Selection logic: VIP > threshold in ANY pairwise comparison
        significant_mask = (self.pair_vips > effective_threshold).any(axis=1)
        self.selected_features = self.pair_vips.index[
            significant_mask
        ].tolist()

        print(
            f'[Feature Selection] Selected {len(self.selected_features)}/{len(X.columns)} '
            f'features with pairwise VIP > {effective_threshold:.2f}'
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns the reduced feature table."""
        if not self.selected_features:
            print('[Warning] No features selected. Returning original table.')
            return X
        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fits and transforms the feature table in one step."""
        return self.fit(X, y).transform(X)
