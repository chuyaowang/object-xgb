# Technical Deep-Dive: Pairwise PLS-DA Feature Selection

This document provides a comprehensive overview of the **Pairwise Partial Least Squares Discriminant Analysis (PLS-DA)** feature selection engine implemented in `object-xgb`.

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

## 3. Comparison: PLS-DA vs. sPLS-DA vs. O-PLS-DA

| Feature | **PLS-DA** (Standard) | **sPLS-DA** (Sparse) | **O-PLS-DA** (Orthogonal) |
| :--- | :--- | :--- | :--- |
| **Primary Goal** | Maximize $X-Y$ Covariance. | Built-in Feature Selection via Sparsity. | Improved Interpretability of Loadings. |
| **Selection Logic** | Post-hoc filtering via VIP scores or loadings. | L1-regularization (Lasso) forces non-essential weights to zero. | Separates "Predictive" variance from "Orthogonal" (noise) variance. |
| **Best Used When...** | You have a moderate number of features and want robust classification. | You have thousands of features (e.g., genomics) and need a very sparse subset. | You need to explain *why* the model is making a decision (Biplot analysis). |
| **Complexity** | Low / Fast. | Moderate (requires tuning sparsity parameters). | High (requires careful rotation of components). |
| **In `object-xgb`** | **Selected** (Standard + Pairwise VIP). | *Planned for future research.* | *Used for Biplot visualizations.* |

---

## 4. Implementation in `object-xgb`

### Group-Aware Extraction
Because PLS-DA depends on a consistent feature set, the `FeatureExtractor` ensures that:
*   If you select one feature in a group (e.g., `raw_mean`), the entire group (e.g., `raw_hist_1...10`) is calculated to maintain the mathematical integrity of the PLS space.
*   The system always outputs a 69-feature table, using `NaN` padding for unselected groups, ensuring the model pipeline doesn't break when moving between different images.

### The XGBoost Hand-off
Once the Pairwise PLS-DA filters the features (typically reducing the set from 69 down to 15-25 high-signal features), the **XGBoost Classifier** takes over. This two-stage approach combines the **discriminative filtering** of PLS-DA with the **non-linear power** of Gradient Boosting.

### Safety Mechanisms
*   **Threshold Capping:** The VIP threshold is capped at `max(VIP) * 0.99`. This ensures that even if a user sets the slider to 5.0, the system will always select at least the single best feature instead of crashing with an empty set.
*   **Balanced Weighting:** During the final XGBoost phase, sample weights are automatically adjusted to prevent the model from ignoring rare object classes.
