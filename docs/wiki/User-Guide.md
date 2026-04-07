# User Guide

This guide describes the workflow for using the `object-rf` plugin to perform object-level classification.

## Workflow Overview

### 1. Object Segmentation
The starting point is often a probability map (e.g., from `napari-rf`).
- **Select Probability/Label Layer**: Choose the layer containing pixel-level probabilities or a binary mask.
- **Automated Segmentation**: The plugin automatically converts probabilities into a foreground mask using `argmax`.
- **Hole Filling**: Small internal holes are filled to ensure object integrity.
- **Intelligent Noise Filtering**: The plugin uses K-Means clustering and a linear SVM on log-transformed areas to automatically identify and remove small false positives caused by pixel-wise predictions. This is triggered only if small objects ($\le 10$ px) are detected.
- **Expansion (Dilation)**: Identified objects are slightly dilated to better capture boundary intensities.
- **Labeling**: Objects are assigned unique, sequential IDs in a new labels layer.

### 2. Feature Extraction
The plugin automatically calculates a comprehensive set of features for each object slice-by-slice:
- **Normalization**: Intensities are clipped to the 0.5-99.5% percentile range to ensure consistency across different imaging sessions.
- **Geometry**: Log Area, Eccentricity, Circularity, and 7 Log-Hu Moments.
- **Intensity Statistics**: Mean, Variance, Skewness, Kurtosis, and a 10-bin normalized histogram.
- **Multi-Layer Filtering**: Intensity features are extracted from the **Raw** image, a **Sobel** (edge) filtered image, and a **Frangi** (tubular) filtered image to capture diverse textures.

### 3. Annotation & Training
To teach the classifier, you must provide ground truth for some objects.
- **Draw Labels**: Click the **"Draw Labels"** button. This automatically adds a new labels layer named **`Object Labels`** with the correct spatial dimensions (stripping channel dimensions if a probability layer is selected).
- **Annotate**: Select the `Object Labels` layer and use the brush tool to paint over representative objects. You only need to label a subset of objects and slices.
- **Train Classifier**: Click **"Train Object Classifier"**. The plugin will extract features *only* for the annotated slices and map your brush strokes to the segmented objects by taking the maximum class ID within each object's mask.

### 4. Application
- **Predict**: Click **"Apply Random Forest"** (or **"Apply RF to All Slices"** for 3D).
- **Memory Efficiency**: For 3D stacks, the plugin processes the stack slice-by-slice, discarding features immediately after prediction to minimize RAM usage.
- **Results**: A new labels layer is added showing the predicted classes.

## UI Features & IO
- **Smart Saving**: All outputs are saved into a subfolder named after your **Intensity Image**, located in the original image's directory.
- **Save Labels**: Exports your manually drawn annotations as a `uint8` TIFF.
- **Save Predictions**: Exports two files for your classification results:
    - `..._class.tif`: An integer map of the most likely class for every pixel.
    - `..._probs.tif`: A floating-point multi-channel map containing the full probability distribution.
- **Save/Load Classifier**: Export and import your trained object-level models (`.joblib`). The default filename is `object_classifier.joblib`.

## Data Shapes & Formats

To ensure compatibility, `object-rf` expects and produces data in the following shapes:

| Component | 2D Shape | 3D Shape | Format |
| :--- | :--- | :--- | :--- |
| **Probability Input** | `(C, Y, X)` | `(Z, C, Y, X)` | Float (0.0 to 1.0) |
| **Label/Mask Input** | `(Y, X)` | `(Z, Y, X)` | Integer |
| **Intensity Source** | `(Y, X)` | `(Z, Y, X)` | Any |
| **Object Labels** | `(Y, X)` | `(Z, Y, X)` | Sequential Integer (1 to N) |
| **Class Predictions** | `(Y, X)` | `(Z, Y, X)` | Integer (User-defined classes) |
| **Probability Output**| `(C, Y, X)` | `(Z, C, Y, X)` | Float32 |

- **C**: Number of classes.
- **Z**: Number of slices in a 3D stack.
- **Y, X**: Height and Width of the image.

### Notes on 3D Shape
In state management and file exports, `object-rf` adheres to the `(Z, C, Y, X)` convention for multi-channel 3D data. This ensures that probability maps can be directly opened as hyperstacks in standard image analysis software like ImageJ/Fiji.
