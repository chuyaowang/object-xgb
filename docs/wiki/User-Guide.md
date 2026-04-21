# User Guide

This guide describes the workflow for using the **object-xgb** plugin to perform object-level classification.

## Workflow Overview

### 1. Object Segmentation
The starting point is often a probability map (e.g., from `napari-rf`).
- **Select Probability/Label Layer**: Choose the layer containing pixel-level probabilities or a binary mask.
- **Automated Segmentation**: The plugin automatically converts probabilities into a foreground mask using `argmax`.
- **Hole Filling**: Small internal holes are filled to ensure object integrity.
- **Intelligent Noise Filtering**: The plugin uses K-Means clustering and a linear SVM on log-transformed areas to automatically identify and remove small false positives caused by pixel-wise predictions. This is triggered only if small objects ($\le 10$ px) are detected.
- **Expansion (Dilation)**: Identified objects are slightly dilated to better capture boundary intensities.
- **Labeling**: Objects are assigned unique, sequential IDs in a new labels layer.

### 2. Annotation & VIP Selection
To teach the classifier, you must provide ground truth for some objects.
- **Draw Labels**: Click the **"Draw Labels"** button. This automatically adds a new labels layer named **`Object Labels`** with the correct spatial dimensions (stripping channel dimensions if a probability layer is selected).
- **Import Labels**: You can drag and drop previously saved manual label files ending in `_object_manual_labels.tif`. The plugin will automatically convert it from an Image layer to a Labels layer, allowing you to seamlessly resume annotating.
- **Annotate**: Select the `Object Labels` layer (or your imported labels layer) and use the brush tool to paint over representative objects. You only need to label a subset of objects and slices.
- **VIP Threshold Slider**: Adjust the **VIP Threshold** (located in the Classifier Controls).
    - **Higher (e.g., 2.0)**: Selects only the "super-features" that strongly separate classes. Best for simple datasets.
    - **Lower (e.g., 0.5)**: Includes more subtle features. Best for complex datasets with high similarity between classes.
    - **Default (1.0)**: A balanced starting point for most biological data.

### 3. Training the Pipeline
- **Train Classifier**: Click **"Train Object Classifier"**.
- **Feature Selection**: The plugin runs a **Pairwise PLS-DA** model to identify the most discriminative features based on your VIP threshold.
- **XGBoost Training**: An XGBoost model is trained on the reduced feature set. It automatically uses **balanced weighting** to ensure small classes are not ignored by the model.

### 4. Prediction
- **Predict**: Click **"Predict Objects"**.
- **Optimization**: The plugin only calculates the feature groups identified as important during the training phase, making prediction much faster on large 3D stacks.
- **Results**: A new image layer **"Object Probabilities"** and a class map are generated.

## Understanding the Report (CSV)
When you click **"Save Features"**, the plugin exports a `full_analysis_report.csv`.

### Column Descriptions
| Column | Description |
| :--- | :--- |
| **slice_id** | The index of the slice (Z) where the object was found. |
| **label** | The unique integer ID of the object in that slice. |
| **true_label** | The manual class ID you painted during training. (0 if unlabeled). |
| **predicted_label** | The class ID predicted by the XGBoost model. |
| **(Features)** | The remaining columns are the 69 morphological and intensity features. |

**Important Note**: If a feature group was not selected by the PLS-DA filter, its columns will contain `NaN` to save computation time while maintaining a consistent table shape.

## UI Features & IO
- **Smart Saving**: All outputs are saved into a subfolder named after your **Intensity Image**, , located in the original image's directory.
- **Save Labels**: Exports your manually drawn annotations as a `uint8` TIFF.
- **Save Predictions**: Exports two files for your classification results:
    - `..._class.tif`: An integer map of the most likely class for every pixel.
    - `..._probs.tif`: A floating-point multi-channel map containing the full probability distribution.
- **Save Features (CSV)**: Exports the `full_analysis_report` for all processed objects to a CSV file. The table includes an auto-populated `true_label` column so you can track which objects were manually annotated during training and which were predicted.
- **Save Model**: Exports the unified PLS-DA + XGBoost pipeline as a `.joblib` file.
- **Reset**: Clears all internal caches and resets the classifier state.

## Data Shapes & Formats

| Component | 2D Shape | 3D Shape | Format |
| :--- | :--- | :--- | :--- |
| **Probability Input** | `(C, Y, X)` | `(Z, C, Y, X)` | Float (0.0 to 1.0) |
| **Label/Mask Input** | `(Y, X)` | `(Z, Y, X)` | Integer |
| **Intensity Source** | `(Y, X)` | `(Z, Y, X)` | Any |
| **Object Labels** | `(Y, X)` | `(Z, Y, X)` | Sequential Integer (1 to N) |
| **Class Predictions** | `(Y, X)` | `(Z, Y, X)` | Integer |
| **Probability Output**| `(C, Y, X)` | `(Z, C, Y, X)` | Float32 |

- **C**: Number of classes.
- **Z**: Number of slices in a 3D stack.
- **Y, X**: Height and Width of the image.

### Notes on 3D Shape
In state management and file exports, `object-xgb` adheres to the `(Z, C, Y, X)` convention for multi-channel 3D data. This ensures that probability maps can be directly opened as hyperstacks in standard image analysis software like ImageJ/Fiji.
