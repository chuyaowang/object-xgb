# Gemini Context: object-xgb

## Project Overview
`object-xgb` is a napari plugin for object-level classification using Random Forest. It bridges pixel-level segmentation and high-level object analysis by extracting morphological and intensity features from segmented objects and training classifiers to categorize them.

### Main Technologies
- **Python**: Core language.
- **napari**: Multi-dimensional image viewer and plugin framework.
- **scikit-image**: Used for segmentation, labeling, and feature extraction.
- **scikit-learn**: Provides base PLS and utility functions.
- **XGBoost**: High-performance gradient boosting for object classification.
- **Qt/qtpy**: GUI framework.
- **joblib**: Model serialization.

### Key Architecture
- **Modular Design**:
    - **`_widget.py`**: Central orchestrator. Handles `true_label` metadata population during training to ensure CSV reports match user actions.
    - **`classifier.py`**: Unified `ObjectClassifier` pipeline managing `PairwisePLSFeatureSelector` (for discriminative filtering), `FeatureAugmentor` (for data synthesis), and `ObjectXGBoostClassifier` (for non-linear classification).
    - **`augmentation.py`**: Scale-aware data augmentation. Implements signal-dependent Gaussian jittering, random scaling, and SMOTE-style interpolation to handle imbalanced classes.
    - **`state.py`**: `ImageStateManager` tracks feature tables and probability caches.
    - **`workers.py`**: Asynchronous generators for segmentation, training, and prediction.
- **Feature Extraction**:
    - **Fixed Schema**: `FeatureExtractor` always produces a 69-feature table (Metadata + Geometry + Hu + Intensity + Texture).
    - **Group-Aware Optimization**: Requesting one feature in a group triggers the calculation of the entire group.
    - **Data Synthesis**: Labeled training data is automatically augmented (2x repeats) and balanced (SMOTE-style) in RAM during training to improve model robustness. Gaussian jitter is signal-dependent (multiplicative) to preserve intensity relationships between classes. Synthetic samples are never exported to disk.
    - **Consistency**: Tables from different images are unified via `reindex` and NaN padding to ensure model pipeline compatibility.

---

## Development Workflow & Preferences

### Git and Source Control
- **Branching**: Use descriptive feature branches (e.g., `feature/object-extraction`).
- **Commit Logic**: Commit logically grouped changes.
- **Commit Message Style**: **Comprehensive and detailed**. Always use the `commit type: commit message` format for commit messages (e.g., `fix: add missing dependencies`, `feat: add new classifier`). Include a summary and a bulleted list of technical improvements.

### Documentation and Wiki
- **"Code Wiki" Preference**: Maintain a technical "code wiki" in `docs/wiki/`.
- **Comprehensiveness**: Integrate new features into the existing context without deleting existing documentation.
- **Technical Detail**: Specify data shapes, feature lists, and architectural flows.

---

## Testing Standards & CI

### Test Organization
- **Logic Tests**: Place core algorithmic and data processing tests in the root `tests/` directory (e.g., `tests/test_logic.py`). These tests **must not** require a `napari.Viewer` or any GUI elements, allowing them to run in headless CLI environments.
- **GUI Tests**: Place tests that require a `napari.Viewer` or Qt widgets in a `tests/gui/` subdirectory (e.g., `tests/gui/test_widget.py`).

### Headless Compatibility
- **Pytest Configuration**: To prevent crashes in environments without a display (like remote SSH or basic CI), the `tests/gui/` directory is excluded from default `pytest` runs via `pyproject.toml` (`norecursedirs = ["gui"]`).
- **Running GUI Tests**: When a display is available, run GUI tests explicitly: `pytest tests/gui/`.

### Updating Testing Setup
- **New Dependencies**: When adding libraries that are imported in the source code, always update `pyproject.toml`'s `dependencies` section. If the library is only needed for testing, add it to `dependency-groups.dev`.
- **CI Synchronization**: Ensure that `tox.ini` and `.github/workflows/test_and_deploy.yml` are updated if new Python versions are supported or if specific system-level dependencies (like OpenGL libs) are required.
- **Mocking**: For logic tests that involve complex `napari` objects, prefer mocking the objects or testing the underlying functions with simple `numpy` arrays to maintain headless compatibility.

---

## Technical Conventions

### Data Integration
- **Upstream Compatibility**: Designed to ingest probability layers produced by `napari-rf`.
- **Feature Robustness**: Handle multi-dimensional images (2D/3D stacks) during feature extraction.
- **State Management**: Persist feature tables associated with label layers to avoid redundant computations.

### Coding Style & Logic
- **Separation of Concerns (Workers)**: Long-running algorithms MUST be placed in `workers.py` as pure functions or generators. They should yield progress and return results without interacting directly with `self.viewer` or Qt elements.
- **Explicit State Management**: Prefer routing all state tracking and cache manipulation through `ImageStateManager` in `state.py`. Avoid "magic number" shape checks (e.g., `ndim == 4`) to infer state.
- **Standardized Terminology**:
    - **`image`**: Refers to the data source or image object.
    - **`slice`**: Refers to a specific 2D plane within a 3D stack.
    - **`layer`**: Reserved specifically for napari UI layer components.
- **Docstrings**: Function docstrings must be detailed and explicit. Always include the purpose of the function, and use standard sections for `Parameters` (including shapes and types) and `Returns`/`Yields` (including precise return types like `list[dict]` or `pd.DataFrame`).
- **Condition Flags**: Use explicit function arguments (e.g., `feature_type="training"`) to communicate intent instead of checking variable properties (like list lengths) to infer logic.
- **Status Reporting**: Provide clear console reports for the lifecycle of operations:
    - **Success/Failure**: Report the outcome of training, prediction, and I/O.
    - **Metadata**: Report when image selection or paths are updated.
    - **I/O Actions**: Explicitly print the target path when saving or loading models and labels.

### Linting & Formatting Standards (Ruff/Pre-commit)
- **Quotes**: Prefer **single quotes** (`'`) for strings unless double quotes are required for nesting.
- **Generators**: Always use `yield from` when yielding over an existing generator or iterable.
- **Unused Variables**: Loop control variables not used within the loop body must be prefixed with an underscore (e.g., `for _idx, row in df.iterrows():`).
- **Exceptions**: **NEVER** use blind `except Exception:`. Always catch specific error types (e.g., `ValueError`, `RuntimeError`, `IOError`).
- **Statement Density**: Do not put multiple statements on a single line (e.g., `if x: return` must be split into two lines).
- **Closure Safety**: When defining a function inside a loop (closure), ensure all loop-dependent variables are explicitly passed as arguments to avoid late-binding artifacts.
- **Whitespace**: No trailing whitespace and exactly one newline at the end of every file.

## Interaction Guideline

- When the users asks for a code change or bug fix. Always analyze the request first, draft a plan, explain the reasoning, and wait for the user's explicit confirmation to begin coding.
