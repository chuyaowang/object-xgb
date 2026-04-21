import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class FeatureAugmentor:
    """
    Utility for performing scale-aware augmentation on tabular features.
    Supports Gaussian jittering, random scaling, and SMOTE-style interpolation.
    """

    def __init__(
        self,
        noise_level: float = 0.05,
        scale_level: float = 0.05,
        dropout_rate: float = 0.05,
        n_repeats: int = 2,
        k_neighbors: int = 5,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        noise_level : float
            Standard deviation of Gaussian noise, relative to each feature's std.
        scale_level : float
            Maximum percentage shift for random scaling (e.g., 0.05 is ±5%).
        dropout_rate : float
            Percentage of features to randomly drop (replace with column mean).
        n_repeats : int
            Number of augmented copies to generate for ALL samples.
        k_neighbors : int
            Number of neighbors for SMOTE-style interpolation.
        random_state : int
            Seed for reproducibility.
        """
        self.noise_level = noise_level
        self.scale_level = scale_level
        self.dropout_rate = dropout_rate
        self.n_repeats = n_repeats
        self.k_neighbors = k_neighbors
        self.rng = np.random.default_rng(random_state)

    def augment(
        self, X: pd.DataFrame, y: pd.Series, balance: bool = False
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Generates augmented copies of the input data.

        Parameters
        ----------
        X : pd.DataFrame
            The feature matrix to augment.
        y : pd.Series
            The target labels for the input data.
        balance : bool
            If True, performs SMOTE-style interpolation to balance classes
            up to the size of the majority class.

        Returns
        -------
        X_aug : pd.DataFrame
            The augmented feature matrix (including the original data).
        y_aug : pd.Series
            The labels corresponding to the augmented matrix.
        """
        all_X = [X.copy()]
        all_y = [y.copy()]

        # 1. Global Augmentation (Repeat for all samples)
        if self.n_repeats > 0:
            X_rep, y_rep = self._apply_noise_and_scale(X, y, self.n_repeats)
            all_X.append(X_rep)
            all_y.append(y_rep)

        # 2. SMOTE-style Balancing (Interpolation for minority classes)
        if balance:
            X_bal, y_bal = self._apply_smote(X, y)
            if not X_bal.empty:
                all_X.append(X_bal)
                all_y.append(y_bal)

        X_aug = pd.concat(all_X, axis=0, ignore_index=True)
        y_aug = pd.concat(all_y, axis=0, ignore_index=True)

        return X_aug, y_aug

    def _apply_noise_and_scale(
        self, X: pd.DataFrame, y: pd.Series, n: int
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Adds signal-dependent jitter, scaling and dropout to n copies of the data."""
        res_X = []
        res_y = []

        # Calculate class means for dropout to preserve class centroids
        class_means = X.groupby(y).transform('mean')

        for _ in range(n):
            X_new = X.copy()

            # 1. Signal-Dependent Gaussian Jittering (Multiplicative)
            # noise = X * N(0, noise_level)
            # This ensures dimmer classes get proportionally smaller noise.
            noise_factors = self.rng.normal(
                0, self.noise_level, size=X_new.shape
            )
            X_new *= 1 + noise_factors

            # 2. Random Scaling (Global feature scaling)
            scaling_factors = self.rng.uniform(
                1 - self.scale_level, 1 + self.scale_level, size=X_new.shape[1]
            )
            X_new *= scaling_factors

            # 3. Class-Aware Feature Dropout
            # Replace features with the mean of their class to prevent drift.
            mask = self.rng.random(size=X_new.shape) < self.dropout_rate
            X_new = X_new.where(~mask, class_means)

            res_X.append(X_new)
            res_y.append(y.copy())

        return pd.concat(res_X, axis=0), pd.concat(res_y, axis=0)

    def _apply_smote(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Performs class balancing via synthetic interpolation."""
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            return pd.DataFrame(), pd.Series(dtype=int)

        max_count = class_counts.max()
        synthetic_X = []
        synthetic_y = []

        for cls, count in class_counts.items():
            if count >= max_count:
                continue

            n_synthetic = max_count - count
            X_cls = X[y == cls].values

            # We need at least 2 samples to interpolate
            if count < 2:
                # Fallback to noise-based augmentation for single-sample classes
                X_noise, y_noise = self._apply_noise_and_scale(
                    X[y == cls], y[y == cls], n_synthetic
                )
                synthetic_X.append(X_noise)
                synthetic_y.append(y_noise)
                continue

            # Find neighbors within the same class
            k = min(self.k_neighbors, count - 1)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(X_cls)
            _, indices = nn.kneighbors(X_cls)

            for _ in range(n_synthetic):
                # Pick a random sample from the class
                idx = self.rng.integers(0, count)
                # Pick a random neighbor (excluding itself)
                neighbor_idx = self.rng.choice(indices[idx, 1:])

                # Interpolate
                diff = X_cls[neighbor_idx] - X_cls[idx]
                new_sample = X_cls[idx] + self.rng.random() * diff

                synthetic_X.append(
                    pd.DataFrame([new_sample], columns=X.columns)
                )
                synthetic_y.append(pd.Series([cls]))

        if not synthetic_X:
            return pd.DataFrame(), pd.Series(dtype=int)

        return pd.concat(synthetic_X, ignore_index=True), pd.concat(
            synthetic_y, ignore_index=True
        )
