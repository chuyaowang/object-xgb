# Pairwise PLS-DA Feature Selection + XGBoost

This document provides a overview of the **Pairwise Partial Least Squares Discriminant Analysis (PLS-DA)** feature selection + XGBoost pipeline implemented in `object-xgb`.

---

## 1. Conceptual Framework

### What is PLS-DA?
Partial Least Squares Discriminant Analysis (PLS-DA) is a supervised version of Principal Component Analysis (PCA). While PCA seeks directions of maximum *variance*, PLS-DA seeks directions that maximize the *covariance* between the feature matrix ($X$) and the class membership matrix ($Y$).

### The Pairwise Strategy
In multi-class biological datasets, global feature selection often fails because a feature might be critical for distinguishing "Class A vs Class B" but completely irrelevant for "Class B vs Class C."

To solve this, `object-xgb` employs a **Pairwise One-vs-One** approach:
1.  The data is split into all possible pairs of classes.
2.  A unique PLS-DA model is trained for each pair.
3.  Feature importance is calculated for every pair independently.
4.  A feature is "selected" if it is highly discriminative for **at least one** pair.

---

## 2. Mathematical Components

### Scores ($T$) and Loadings ($P$)
*   **Scores ($T$):** The coordinates of the objects in the new, low-dimensional PLS space. These are the "latent variables" used for visualization and initial classification.
*   **Loadings ($P$):** The weights assigned to each original feature (e.g., area, intensity) to create the scores. High absolute loadings indicate that a feature is a strong driver of that specific PLS component.

### VIP Scores (Variable Importance in Projection)
The VIP score is the primary metric used for feature selection in this plugin. It summarizes the contribution of a feature across all calculated components, weighted by the amount of $Y$-variance explained by each component.

**The Formula:**
$$VIP_{j} = \sqrt{p \cdot \frac{\sum_{h=1}^H SS_h \cdot (w_{hj} / ||w_h||)^2}{\sum_{h=1}^H SS_h}}$$

*   **$p$**: Total number of features (69 in our case).
*   **$SS_h$**: The sum of squares explained by component $h$ (the "predictive power" of that component).
*   **$w_{hj}$**: The weight of feature $j$ on component $h$.
*   **$H$**: Number of components (fixed to 2 for optimal pairwise separation).

**Interpretation Guidelines:**
*   **VIP > 1.0**: The feature is a significant contributor to class separation.
*   **VIP < 0.8**: The feature is likely noise or redundant.
*   **0.8 < VIP < 1.0**: Marginal importance; depends on the dataset complexity.

---

## 3. Implementation in `object-xgb`

### Group-Aware Extraction
Because PLS-DA depends on a consistent feature set, the `FeatureExtractor` ensures that:
*   If you select one feature in a group (e.g., `raw_mean`), the entire group (e.g., `raw_hist_1...10`) is calculated to maintain the mathematical integrity of the PLS space.
*   The system always outputs a 69-feature table, using `NaN` padding for unselected groups, ensuring the model pipeline doesn't break when moving between different images.

### The XGBoost Hand-off
Once the Pairwise PLS-DA filters the features (typically reducing the set from 69 down to 15-25 high-signal features), the **XGBoost Classifier** takes over. This two-stage approach combines the **discriminative filtering** of PLS-DA with the **non-linear power** of Gradient Boosting.

### Safety Mechanisms
*   **Threshold Capping:** The VIP threshold is capped at `max(VIP) * 0.99`. This ensures that even if a user sets the slider to 5.0, the system will always select at least the single best feature instead of crashing with an empty set.
*   **Balanced Weighting:** During the final XGBoost phase, sample weights are automatically adjusted to prevent the model from ignoring rare object classes.

---

## 4. Automated Data Augmentation & Balancing

To improve model robustness and handle biological datasets with limited or imbalanced annotations, `object-xgb` automatically synthesizes training data in RAM during the training phase.

### Scale-Aware Augmentation
Tabular features have vastly different scales (e.g., area in thousands, intensity in fractions). The `FeatureAugmentor` applies augmentations using signal-dependent logic:

1.  **Signal-Dependent Gaussian Jittering**: Adds multiplicative noise: $X_{new} = X \times (1 + \mathcal{N}(0, \text{noise\_level}))$. This ensures that dim objects receive proportionally smaller noise than bright objects, preserving the intensity profile of different classes.
2.  **Random Scaling**: Multiplies features by a random factor ($0.95 - 1.05$) to simulate global intensity or size variations (calibration errors).
3.  **Class-Aware Feature Dropout**: Randomly replaces $5\%$ of features with the **mean value of that feature within its specific class**. This prevents feature drift and ensures synthetic samples remain representative of their biological category.

### SMOTE-style Balancing
If one class has significantly fewer labels than others, the system uses synthetic interpolation:
*   **Mechanism**: It identifies the nearest neighbors of a minority sample **within the same class** and generates a new point along the line connecting them.
*   **Benefit**: This populates the "decision space" of rare biological events (like "Invaded") using the natural variance found in the real data.
*   **Fallback**: If only a single sample exists for a class, the system falls back to noise-based jittering for that class.

**Note**: Augmented samples are used strictly for model fitting. They are never stored in the `ImageStateManager` or exported to the final CSV analysis report, ensuring your raw measurements remain untainted.
