from typing import TYPE_CHECKING, Any

import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage import measure

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

        # --- UI Components ---
        self.setLayout(QVBoxLayout())

        # 1. Image Layer Selection (Probabilities or masks)
        self.layout().addWidget(QLabel('Select Image/Probability Layer:'))
        self.layer_combo = QComboBox()
        self.layer_combo.setToolTip(
            'Select the image or multi-channel probability layer to segment.'
        )
        self.layer_combo.currentIndexChanged.connect(self._on_layer_change)
        self.layout().addWidget(self.layer_combo)

        self.btn_segment = QPushButton('Segment Objects')
        self.btn_segment.setToolTip(
            'Generate unique object labels from the selected layer.'
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

    def _init_image_state(self, image):
        """Initialize the state dictionary for a specific image layer."""
        if image is None or image in self.image_states:
            return

        path = None
        source = getattr(image, 'source', None)
        raw_path = getattr(source, 'path', None)
        if raw_path:
            path = str(
                raw_path[0]
                if isinstance(raw_path, (list, tuple))
                else raw_path
            )

        self.image_states[image] = {
            'data': image.data,
            'ndim': image.data.ndim,
            'name': image.name,
            'path': path,
            'objects': None,  # Identified unique labels
            'labeled_slices': [],  # Indices for 3D stacks
            'features': None,  # [object, feature] matrix
            'training_probabilities': None,
            'prediction_probabilities': None,
        }
        print(f'[Object RF] Initialized state for: {image.name}')

    def _on_layer_change(self, event=None):
        """Update dropdown and manage state transition between image layers."""
        import napari

        # 1. Update Dropdown
        current_selection = self.layer_combo.currentText()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()

        image_layers = [
            layer
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]
        self.layer_combo.addItems([layer.name for layer in image_layers])

        # Restore selection
        for i, layer in enumerate(image_layers):
            if layer.name == current_selection:
                self.layer_combo.setCurrentIndex(i)
                break
        else:
            if image_layers:
                self.layer_combo.setCurrentIndex(0)
        self.layer_combo.blockSignals(False)

        # 2. Handle State Migration
        active_image = self.get_selected_layer()
        if active_image != self._current_image:
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

            self._current_image = active_image
            self._init_image_state(active_image)

        # 3. Enable/Disable UI
        has_layers = active_image is not None
        self.btn_segment.setEnabled(has_layers)

        state = self.image_states.get(active_image)
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

    def segment_objects(self):
        """
        Segment objects from the selected layer.
        - If Probabilities (Float): Uses argmax to find the most likely class per pixel
          (axis 1 for 4D, axis 0 for 3D), treats classes > 0 as foreground.
        - If Labels/Mask (Integer): Treats all values > 0 as foreground.
        Finally, assigns unique object IDs to all identified foreground objects.
        """
        layer = self.get_selected_layer()
        if layer is None:
            return

        state = self.image_states.get(layer)
        if state is None:
            return

        data = layer.data
        is_float = np.issubdtype(data.dtype, np.floating)

        if is_float:
            # Probability-to-Class logic from napari-rf
            # 2D multi-channel: (C, Y, X) -> axis 0
            # 3D multi-channel: (Z, C, Y, X) -> axis 1
            argmax_axis = 1 if data.ndim == 4 else 0
            class_map = np.argmax(data, axis=argmax_axis)
            foreground = class_map > 0
        else:
            # Already labels or binary mask
            foreground = data > 0

        # Assign unique IDs to discrete objects
        labels = measure.label(foreground)
        state['objects'] = labels

        new_name = f'{layer.name}_objects'
        if new_name in self.viewer.layers:
            self.viewer.layers[new_name].data = labels
        else:
            self.viewer.add_labels(labels, name=new_name)

        self.btn_extract.setEnabled(True)

    def extract_features(self):
        """Calculate features for current object set."""
        state = self.image_states.get(self._current_image)
        if state is None or state['objects'] is None:
            return

        print(f"Extracting features for '{state['name']}'...")
        # (Mock feature calculation)
        # state['features'] = calculate_props(state['objects'], self._current_image.data)

        self.btn_train.setEnabled(True)

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
