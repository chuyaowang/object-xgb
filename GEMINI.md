# Gemini Context: object-rf

## Project Overview
`object-rf` is a napari plugin for object-level classification using Random Forest. It bridges pixel-level segmentation and high-level object analysis by extracting morphological and intensity features from segmented objects and training classifiers to categorize them.

### Main Technologies
- **Python**: Core language.
- **napari**: Multi-dimensional image viewer and plugin framework.
- **scikit-image**: Used for segmentation (thresholding), labeling (`measure.label`), and feature extraction (`measure.regionprops`).
- **scikit-learn**: Provides the `RandomForestClassifier` for object-level classification.
- **Qt/qtpy**: GUI framework.
- **joblib**: Model serialization.

### Key Architecture
- **`ObjectWidget` (`src/object_rf/_widget.py`)**: (Planned) The main GUI for controlling thresholding, feature extraction, and training.
- **Feature Extraction**: Leverages `regionprops_table` to generate a tabular representation of objects for machine learning.
- **Workflow**:
    1.  Convert Probability maps (from `napari-rf`) into Labels.
    2.  Compute geometrical (area, perimeter) and intensity (mean, max) features.
    3.  User-guided training (annotating specific labels with classes).
    4.  Full-stack object classification.

---

## Development Workflow & Preferences

### Git and Source Control
- **Branching**: Use descriptive feature branches (e.g., `feature/object-extraction`).
- **Commit Logic**: Commit logically grouped changes.
- **Commit Message Style**: **Comprehensive and detailed**. Include a summary and a bulleted list of technical improvements.

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
- **Explicit State Management**: Prefer centralized state dictionaries (`image_states`) to track data and caches. Avoid "magic number" shape checks (e.g., `ndim == 4`) to infer state.
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
