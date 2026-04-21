import numpy as np
import pandas as pd

from object_xgb.augmentation import FeatureAugmentor
from object_xgb.classifier import ObjectClassifier
from object_xgb.feature_selection import PairwisePLSFeatureSelector
from object_xgb.xgboost_classifier import ObjectXGBoostClassifier


def test_pls_feature_selector():
    # Create synthetic data with 10 features, only first 2 are useful
    n_samples = 40
    n_features = 10
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feat_{i}' for i in range(n_features)],
    )

    # Target y depends only on feat_0 and feat_1
    y = pd.Series(np.zeros(n_samples))
    y[X['feat_0'] + X['feat_1'] > 0] = 1
    y[X['feat_0'] + X['feat_1'] <= 0] = 2

    selector = PairwisePLSFeatureSelector(threshold=0.5)
    selector.fit_transform(X, y)

    # Should select at least some features
    assert len(selector.selected_features) > 0
    # Useful features should likely be in selected_features
    assert (
        'feat_0' in selector.selected_features
        or 'feat_1' in selector.selected_features
    )


def test_xgboost_classifier_basic():
    # Simple binary classification
    X = pd.DataFrame(
        np.random.randn(20, 5), columns=[f'f{i}' for i in range(5)]
    )
    y = pd.Series([1] * 10 + [2] * 10)

    clf = ObjectXGBoostClassifier()
    clf.train(X, y)

    preds = clf.predict(X)
    assert len(preds) == 20
    assert set(preds).issubset({1, 2})


def test_object_classifier_pipeline():
    # End-to-end test
    n_samples = 50
    X = pd.DataFrame(
        np.random.randn(n_samples, 10), columns=[f'f{i}' for i in range(10)]
    )
    y = pd.Series(np.random.choice([1, 2, 3], size=n_samples))

    pipeline = ObjectClassifier(threshold=0.1)
    pipeline.train(X, y)

    assert len(pipeline.selected_features) > 0

    preds = pipeline.predict(X)
    assert len(preds) == n_samples
    assert set(preds).issubset({1, 2, 3})

    probas = pipeline.predict_proba(X)
    assert probas.shape == (n_samples, 3)


def test_full_report_generation():
    n_samples = 10
    X = pd.DataFrame(
        np.random.randn(n_samples, 5), columns=[f'f{i}' for i in range(5)]
    )
    y = pd.Series([1, 2] * 5)
    original_df = pd.DataFrame(
        {'slice_id': np.zeros(n_samples), 'label': np.arange(1, n_samples + 1)}
    )

    pipeline = ObjectClassifier()
    pipeline.train(X, y)

    report = pipeline.get_report(X, y, original_df)

    assert isinstance(report, pd.DataFrame)
    assert all(
        col in report.columns
        for col in ['slice_id', 'label', 'true_label', 'predicted_label']
    )
    assert len(report) == n_samples


def test_augmentation_logic():
    n_samples = 10
    # Features with different scales
    X = pd.DataFrame(
        {
            'small': np.random.normal(0, 1, n_samples),
            'large': np.random.normal(0, 1000, n_samples),
            'constant': np.ones(n_samples),
        }
    )
    y = pd.Series([1] * 5 + [2] * 5)

    n_repeats = 2
    augmentor = FeatureAugmentor(n_repeats=n_repeats, random_state=42)
    X_aug, y_aug = augmentor.augment(X, y)

    # Should have original + n_repeats * original
    assert len(X_aug) == n_samples * (1 + n_repeats)
    assert len(y_aug) == len(X_aug)

    # Verify scale-awareness (rough check)
    # The 'large' feature noise should be much larger than 'small' feature noise
    # We compare the std of the augmented rows (excluding the first n_samples)
    noise_part = X_aug.iloc[n_samples:]
    assert noise_part['large'].std() > noise_part['small'].std() * 10


def test_classifier_with_augmentation():
    n_samples = 20
    X = pd.DataFrame(
        np.random.randn(n_samples, 5), columns=[f'f{i}' for i in range(5)]
    )
    y = pd.Series([1] * 10 + [2] * 10)

    # Train with augmentation
    pipeline = ObjectClassifier(
        use_augmentation=True, augmentation_params={'n_repeats': 1}
    )
    pipeline.train(X, y)

    preds = pipeline.predict(X)
    assert len(preds) == n_samples
    assert set(preds).issubset({1, 2})
