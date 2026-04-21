import numpy as np
import pandas as pd

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
