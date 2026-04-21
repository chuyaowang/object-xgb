# object-xgb

[![License MIT](https://img.shields.io/pypi/l/object-xgb.svg?color=green)](https://github.com/chuyaowang/object-xgb/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/object-xgb.svg?color=green)](https://pypi.org/project/object-xgb)
[![Python Version](https://img.shields.io/pypi/pyversions/object-xgb.svg?color=green)](https://python.org)
[![tests](https://github.com/chuyaowang/object-xgb/workflows/tests/badge.svg)](https://github.com/chuyaowang/object-xgb/actions)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/object-xgb)](https://napari-hub.org/plugins/object-xgb)

**object-xgb** is a napari plugin for **object-level classification** using a high-performance **PLS-DA + XGBoost** pipeline. It bridges the gap between pixel-level segmentation and high-level biological analysis by transforming segmented objects into rich feature vectors and classifying them using advanced machine learning.

---

## Key Features

- **Optimized Feature Extraction**: Extract 69 geometrical, intensity, and texture features (including GLCM, LBP, Hu Moments, and Frangi/Sobel filters).
- **Pairwise PLS-DA Filtering**: Automated feature selection using pairwise VIP (Variable Importance in Projection) scores to isolate the most discriminative signals.
- **XGBoost Classification**: Robust, multi-class classification with automated class-imbalance handling (balanced weighting).
- **Automated Data Augmentation**: Automatically generates synthetic training samples via scale-aware Gaussian jittering, random scaling, and SMOTE-style interpolation to improve model generalization and handle rare biological classes.
- **Group-Aware Computation**: Intelligent feature calculation that avoids redundant processing by gating computations based on user selection.
- **Detailed Reporting**: Export comprehensive CSV reports containing object metadata, ground truth, predictions, and the full feature matrix.
- **3D Stack Support**: Fully compatible with large 2D/3D image stacks, processing data slice-by-slice to maintain low RAM overhead.

---

## Installation

You can install `object-xgb` via [pip]:

```bash
pip install object-xgb
```

To install the required XGBoost library:

```bash
pip install xgboost
```

## Usage

1.  **Segment**: Convert your pixel-level probability maps (e.g., from `napari-rf`) into discrete object labels via thresholding and connected component analysis.
2.  **Annotate**: Paint manual labels for a few objects in the "Object Labels" layer to serve as training examples.
3.  **Train**: Set your **VIP Threshold** (default 1.0) and click **Train**. The plugin will automatically select the best features using Pairwise PLS-DA and train an XGBoost model.
4.  **Predict**: Apply the trained model to the entire 3D stack.
5.  **Export**: Save your results as class maps, probability maps, or a full CSV analysis report.

---

## Technology Stack

- **napari**: Interactive visualization.
- **scikit-image**: Segmentation and feature extraction.
- **scikit-learn**: PLS-DA and model utilities.
- **XGBoost**: High-performance gradient boosting for classification.
- **pandas**: Tabular data management.

---

## Documentation

For detailed technical guides, mathematical deep-dives (including PLS-DA VIP scores), and API references, please visit our **[GitHub Wiki](https://github.com/chuyaowang/object-xgb/wiki)**.

---

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure the coverage at least stays the same before you submit a pull request.

---

## License

Distributed under the terms of the [MIT] license, "object-xgb" is free and open-source software.

[napari]: https://github.com/napari/napari
[MIT]: http://opensource.org/licenses/MIT
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
