# API Reference

This page provides technical details for the primary classes and functions of the **object-xgb** plugin.

## `object_xgb.classifier.ObjectClassifier`
The unified pipeline orchestrator.

### `ObjectClassifier(vip_threshold=1.0)`
- **`selector`**: Instance of `PairwisePLSFeatureSelector`.
- **`model`**: Instance of `ObjectXGBoostClassifier`.
- **`train(X, y)`**:
    1. Fits the PLS-DA selector to identify important features.
    2. Filters the feature matrix $X$ based on the VIP threshold.
    3. Fits the XGBoost model to the reduced data.
- **`predict(X)`**:
    1. Transforms $X$ using the cached `selected_features`.
    2. Returns integer class predictions.
- **`get_report(X, y, original_df)`**: Returns a DataFrame containing metadata, ground truth, and predictions.

---

## `object_xgb.feature_selection.PairwisePLSFeatureSelector`
Custom feature selector based on Variable Importance in Projection (VIP) scores.

### `PairwisePLSFeatureSelector(threshold=1.0)`
- **`fit(X, y)`**: Calculates pairwise VIP scores for all class combinations and identifies the most discriminative features.
- **`transform(X)`**: Filters the input DataFrame to include only the selected feature columns.
- **`selected_features`**: List of strings containing the names of the chosen features.

---

## `object_xgb.xgboost_classifier.ObjectXGBoostClassifier`
Wrapper for the XGBoost gradient boosting engine.

### `ObjectXGBoostClassifier(**kwargs)`
- **`train(X, y)`**:
    - Maps integer labels to 0-indexed values.
    - Calculates balanced sample weights using `compute_sample_weight`.
    - Fits the `xgb.XGBClassifier`.
- **`predict(X)`**: Returns predicted integer class labels (remapped from 0-indexed internal values).

---

## `object_xgb.feature_extraction.FeatureExtractor`
Core logic for calculating object properties slice-by-slice.

### `generate_features(label_image, intensity_image, selected_features=None)`
- **Parameters**:
    - `label_image`: (Y, X) or (Z, Y, X) numpy array.
    - `intensity_image`: Source image for intensity and texture features.
    - `selected_features` (Optional): List of specific features to calculate. If provided, the extractor will skip unneeded feature groups.
- **Output**: A generator yielding progress updates and finally a pandas DataFrame with a fixed 69-feature schema.

---

## `object_xgb._widget.ObjectWidget`
The main Qt GUI interface and state manager.

### Key Methods
- `segment_objects()`: Orchestrates the segmentation pipeline (Argmax -> Filter -> Dilate -> Relabel).
- `train_classifier()`: Fits the unified PLS-DA + XGBoost pipeline.
- `predict_objects()`: Executes the memory-efficient slice-by-slice inference workflow using the optimized feature extractor.
- `save_features()`: Exports the full analysis report using the `true_label` metadata column.
