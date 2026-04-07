# API Reference

This page provides technical details for the primary classes and functions of the `object-rf` plugin.

## `object_rf.classifier.ObjectClassifier`
Wrapper for `sklearn.ensemble.RandomForestClassifier`.

### `ObjectClassifier(n_estimators=100, random_state=42)`
- **`train(X, y)`**: Fits the classifier to object-level features.
- **`predict(X)`**: Returns integer class predictions.
- **`predict_proba(X)`**: Returns class probabilities.

---

## `object_rf.feature_extraction.FeatureExtractor`
Core logic for calculating object properties slice-by-slice.

### `generate_features(label_image, intensity_image=None, indices=None)`
- **Parameters**:
    - `label_image`: (Y, X) or (Z, Y, X) numpy array.
    - `intensity_image` (Optional): Source image for intensity and texture features.
    - `indices` (Optional): List of slice indices to process (for memory-efficient 3D workflows).
- **Output**: A generator yielding progress updates and finally a pandas DataFrame.

---

## `object_rf._widget.ObjectWidget`
The main Qt GUI interface and state manager.

### Key Methods
- `segment_objects()`: Orchestrates the segmentation pipeline (Argmax -> Filter -> Dilate -> Relabel).
- `train_classifier()`: Collects training samples via spatial intersection and fits the model.
- `predict_objects()`: Executes the memory-efficient slice-by-slice inference workflow.
- `save_predictions()` / `save_labels()`: Exports results to subfolders named after the intensity image.
