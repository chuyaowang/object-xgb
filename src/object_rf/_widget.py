from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from napari.qt.threading import thread_worker
from napari.utils import progress
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import binary_fill_holes
from skimage import measure, morphology, segmentation
from sklearn.cluster import KMeans
from sklearn.svm import SVC

from .feature_extraction import FeatureExtractor

if TYPE_CHECKING:
    import napari


class ObjectWidget(QWidget):
    def __init__(self, viewer: 'napari.viewer.Viewer'):
        super().__init__()
        self.viewer = viewer

        # State management: Dictionary holding data and caches for each image layer
        # Key: napari.layers.Image object, Value: dict of state
        self.image_states: dict[napari.layers.Image, dict[str, Any]] = {}
        self._current_image = None
        self.clf = None  # Future RandomForestClassifier
        self._clf_ready = False
        self.feature_extractor = FeatureExtractor()

        # --- UI Components ---
        self.setLayout(QVBoxLayout())

        # 1. Image Layer Selection (Probabilities or masks)
        self.layout().addWidget(QLabel('Select Probability/Label Layer:'))
        self.layer_combo = QComboBox()
        self.layer_combo.setToolTip(
            'Select the image (probabilities) or label layer to segment.'
        )
        self.layer_combo.currentIndexChanged.connect(self._on_layer_change)
        self.layout().addWidget(self.layer_combo)

        # 2. Original Image Selection (Intensity Source)
        self.layout().addWidget(QLabel('Select Original Intensity Image:'))
        self.image_combo = QComboBox()
        self.image_combo.setToolTip(
            'Select the original image to use as the intensity source for features.'
        )
        self.layout().addWidget(self.image_combo)

        self.btn_segment = QPushButton('Segment Objects')
        self.btn_segment.setToolTip(
            'Generate unique object labels and automatically filter noise.'
        )
        self.btn_segment.clicked.connect(self.segment_objects)
        self.layout().addWidget(self.btn_segment)

        # 3. Feature Extraction
        self.btn_extract = QPushButton('Extract Features')
        self.btn_extract.setToolTip(
            'Calculate geometrical and intensity features for each segmented object.'
        )
        self.btn_extract.clicked.connect(self.extract_features)
        self.btn_extract.setDisabled(True)
        self.layout().addWidget(self.btn_extract)

        # 4. Training & Classification
        self.btn_train = QPushButton('Train Object Classifier')
        self.btn_train.clicked.connect(self.train_classifier)
        self.btn_train.setDisabled(True)
        self.layout().addWidget(self.btn_train)

        self.btn_predict = QPushButton('Classify Objects')
        self.btn_predict.clicked.connect(self.predict_objects)
        self.btn_predict.setDisabled(True)
        self.layout().addWidget(self.btn_predict)

        # 5. IO
        self.btn_save_model = QPushButton('Save Object Model')
        self.btn_save_model.clicked.connect(self.save_model)
        self.btn_save_model.setDisabled(True)
        self.layout().addWidget(self.btn_save_model)

        self.btn_load_model = QPushButton('Load Object Model')
        self.btn_load_model.clicked.connect(self.load_model)
        self.layout().addWidget(self.btn_load_model)

        # 6. Reset
        self.btn_reset = QPushButton('Reset All')
        self.btn_reset.clicked.connect(self.reset_all)
        self.layout().addWidget(self.btn_reset)

        # Connect layer events to keep the dropdown updated
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)
        self._on_layer_change()

    def _init_image_state(self, layer):
        """Initialize the state dictionary for a specific layer (Image or Labels)."""
        if layer is None or layer in self.image_states:
            return

        import napari

        path = None
        source = getattr(layer, 'source', None)
        raw_path = getattr(source, 'path', None)
        if raw_path:
            path = str(
                raw_path[0]
                if isinstance(raw_path, (list, tuple))
                else raw_path
            )

        # Determine layer type and original image dimensionality
        data = layer.data
        if isinstance(layer, napari.layers.Image):
            if np.issubdtype(data.dtype, np.floating):
                layer_type = 'probabilities'
                orig_ndim = data.ndim - 1
            else:
                layer_type = 'mask'
                orig_ndim = data.ndim
        else:
            layer_type = 'mask'
            orig_ndim = data.ndim

        self.image_states[layer] = {
            'data': data,
            'ndim': data.ndim,
            'orig_ndim': orig_ndim,
            'layer_type': layer_type,
            'name': layer.name,
            'path': path,
            'objects': None,  # Filtered + dilated label image
            'labeled_slices': [],  # Indices for 3D stacks
            'features': None,  # pd.DataFrame
            'training_probabilities': None,
            'prediction_probabilities': None,
        }
        print(
            f'[Object RF] Initialized state for: {layer.name} '
            f'(Type: {layer_type}, Orig NDIM: {orig_ndim})'
        )

    def _on_layer_change(self, event=None):
        """Update dropdowns and manage state transition between image layers."""
        import napari

        # 1. Update Dropdowns
        current_layer_name = self.layer_combo.currentText()
        current_image_name = self.image_combo.currentText()

        self.layer_combo.blockSignals(True)
        self.image_combo.blockSignals(True)

        self.layer_combo.clear()
        self.image_combo.clear()

        image_layers = [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]
        label_layers = [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]

        # layer_combo shows both Image and Labels
        all_candidate_layers = image_layers + label_layers
        self.layer_combo.addItems(
            [layer.name for layer in all_candidate_layers]
        )

        # image_combo shows only Image
        self.image_combo.addItems([layer.name for layer in image_layers])

        # Restore selection
        for i, layer in enumerate(all_candidate_layers):
            if layer.name == current_layer_name:
                self.layer_combo.setCurrentIndex(i)
                break
        else:
            if all_candidate_layers:
                self.layer_combo.setCurrentIndex(0)

        for i, layer in enumerate(image_layers):
            if layer.name == current_image_name:
                self.image_combo.setCurrentIndex(i)
                break
        else:
            if image_layers:
                self.image_combo.setCurrentIndex(0)

        self.layer_combo.blockSignals(False)
        self.image_combo.blockSignals(False)

        # 2. Handle State Migration
        active_layer = self.get_selected_layer()
        if active_layer != self._current_image:
            if (
                self._current_image
                and self._current_image in self.image_states
            ):
                state = self.image_states[self._current_image]
                if state['features'] is not None:
                    reply = QMessageBox.question(
                        self,
                        'Clear Caches?',
                        f"Switching image. Clear features and probabilities for '{state['name']}'?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No,
                    )
                    if reply == QMessageBox.Yes:
                        del self.image_states[self._current_image]

            self._current_image = active_layer
            self._init_image_state(active_layer)

        # 3. Enable/Disable UI
        has_layers = active_layer is not None
        self.btn_segment.setEnabled(has_layers)

        state = self.image_states.get(active_layer)
        has_objects = state is not None and state['objects'] is not None
        self.btn_extract.setEnabled(has_objects)

        has_features = state is not None and state['features'] is not None
        self.btn_train.setEnabled(has_features)
        self.btn_predict.setEnabled(self._clf_ready and has_features)

    def get_selected_layer(self):
        name = self.layer_combo.currentText()
        if name in self.viewer.layers:
            return self.viewer.layers[name]
        return None

    def get_intensity_layer(self):
        name = self.image_combo.currentText()
        if name in self.viewer.layers:
            return self.viewer.layers[name]
        return None

    def segment_objects(self):
        """
        Segment objects from the selected layer.
        - If Probabilities: Uses argmax to find the most likely class per pixel
          (axis 1 for 4D, axis 0 for 3D), treats classes > 0 as foreground.
        - If Mask: Treats all values > 0 as foreground.
        Finally, assigns unique object IDs to all identified foreground objects.
        1. Argmax (if probs) or Threshold.
        2. Hole filling.
        3. Initial labeling.
        4. Automatic Size Filtering using K-Means and SVM on log-areas.
        5. Dilation.
        6. Sequential reindexing.
        """
        layer = self.get_selected_layer()
        if layer is None:
            return

        # Ensure we have state for the active layer
        self._init_image_state(layer)
        state = self.image_states.get(layer)
        if state is None:
            return

        self.btn_segment.setEnabled(False)

        @thread_worker
        def _segment_worker():
            data = layer.data
            orig_ndim = state['orig_ndim']

            # 1. Pixel-wise segmentation
            if state['layer_type'] == 'probabilities':
                argmax_axis = 1 if orig_ndim == 3 else 0
                class_map = np.argmax(data, axis=argmax_axis)
                foreground = class_map > 0
            else:
                foreground = data > 0

            # 2. Hole filling
            if orig_ndim == 3:
                for z in range(foreground.shape[0]):
                    foreground[z] = binary_fill_holes(foreground[z])
            else:
                foreground = binary_fill_holes(foreground)

            # 3. Initial Labeling
            raw_labels = measure.label(foreground)
            props = measure.regionprops(raw_labels)
            if not props:
                return raw_labels, pd.DataFrame()

            areas = np.array([p.area for p in props]).reshape(-1, 1)
            log_areas = np.log10(areas)

            # 4. Automated Thresholding (K-Means + SVM)
            if len(props) > 1:
                # Group into 2 clusters: Noise vs. Signal
                kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
                cluster_labels = kmeans.fit_predict(log_areas)

                # Train SVM to find the optimal decision boundary (threshold)
                svm = SVC(kernel='linear', C=1.0)
                svm.fit(log_areas, cluster_labels)

                # The decision boundary is where w*x + b = 0 => x = -b/w
                w = svm.coef_[0]
                b = svm.intercept_[0]
                log_threshold = -b / w
                threshold = 10**log_threshold

                # Ensure threshold makes sense (between min and max area)
                # If SVM fails to find a good boundary, default to cluster separation
                if not (np.min(areas) < threshold < np.max(areas)):
                    # Fallback: midpoint between cluster centers
                    centers = 10 ** kmeans.cluster_centers_.flatten()
                    threshold = np.mean(centers)

                # Convert threshold to scalar for printing and comparison
                threshold = float(np.squeeze(threshold))

                print(
                    f'[Object RF] Auto-detected size threshold: {threshold:.2f}'
                )

                # Filter noise
                keep_labels = [p.label for p in props if p.area >= threshold]
            else:
                # Only one object, keep it
                keep_labels = [p.label for p in props]

            mask = np.isin(raw_labels, keep_labels)
            filtered_labels = np.where(mask, raw_labels, 0)

            # 5. Dilation (AFTER filtering)
            if orig_ndim == 3:
                footprint = morphology.ball(1)
                processed_labels = morphology.dilation(
                    filtered_labels, footprint=footprint
                )
            else:
                footprint = morphology.disk(1)
                processed_labels = morphology.dilation(
                    filtered_labels, footprint=footprint
                )

            # 6. Relabel sequentially
            final_labels, _, _ = segmentation.relabel_sequential(
                processed_labels
            )

            # Initial features (area)
            final_props = measure.regionprops(final_labels)
            features = pd.DataFrame(
                [{'label': p.label, 'area': p.area} for p in final_props]
            )

            return final_labels, features

        def _on_done(result):
            final_labels, features = result
            state['objects'] = final_labels
            state['features'] = features

            new_name = f'{state["name"]}_objects'
            if new_name in self.viewer.layers:
                self.viewer.layers[new_name].data = final_labels
            else:
                self.viewer.add_labels(final_labels, name=new_name)

            self.btn_segment.setEnabled(True)
            self.btn_extract.setEnabled(True)
            self.btn_train.setEnabled(not features.empty)
            print(
                f'[Object RF] Automatically filtered to {len(features)} objects.'
            )

        worker = _segment_worker()
        worker.returned.connect(_on_done)
        worker.start()

    def extract_features(self):
        """Calculate intensity features using a thread worker."""
        state = self.image_states.get(self._current_image)
        if state is None or state['objects'] is None:
            return

        intensity_layer = self.get_intensity_layer()
        if intensity_layer is None:
            print('No intensity image selected.')
            return

        intensity_data = intensity_layer.data
        print(
            f"Extracting features using intensity from '{intensity_layer.name}'..."
        )

        self.btn_extract.setEnabled(False)
        pbar = progress(desc='Extracting Object Features')

        @thread_worker
        def _extract_worker():
            gen = self.feature_extractor.generate_features(
                state['objects'],
                intensity_image=intensity_data,
            )
            try:
                while True:
                    yield next(gen)
            except StopIteration:
                pass

        def _on_yielded(val):
            if isinstance(val, tuple):
                curr, total, desc = val
                pbar.total = total
                pbar.n = curr
                pbar.set_description(desc)
                pbar.refresh()
            else:
                # Final yield is the full DataFrame (intensity features)
                new_features = val
                if state['features'] is not None:
                    # Merge intensity features into the existing (area/label) DataFrame
                    state['features'] = pd.merge(
                        state['features'], new_features, on='label', how='left'
                    )
                else:
                    state['features'] = new_features

        def _on_finished():
            pbar.close()
            self.btn_extract.setEnabled(True)
            if state['features'] is not None:
                print(
                    f'Extracted all features for {len(state["features"])} objects.'
                )
                self.btn_train.setEnabled(True)
            else:
                print('Feature extraction failed.')

        worker = _extract_worker()
        worker.yielded.connect(_on_yielded)
        worker.finished.connect(_on_finished)
        worker.start()

    def train_classifier(self):
        """Placeholder for training."""
        print('Training object classifier...')
        self._clf_ready = True
        self.btn_predict.setEnabled(True)
        self.btn_save_model.setEnabled(True)

    def predict_objects(self):
        """Placeholder for prediction."""
        print('Predicting object classes...')

    def save_model(self):
        print('Saving model...')

    def load_model(self):
        print('Loading model...')

    def reset_all(self):
        """Reset internal state to default."""
        self.clf = None
        self.image_states.clear()
        self._current_image = None
        self._clf_ready = False
        self._on_layer_change()
