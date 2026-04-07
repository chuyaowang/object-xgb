from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from joblib import dump, load
from napari.qt.threading import thread_worker
from napari.utils import progress
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import binary_fill_holes
from skimage import measure, morphology, segmentation
from skimage.io import imsave
from sklearn.cluster import KMeans
from sklearn.svm import SVC

from .classifier import ObjectClassifier
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
        self.clf = ObjectClassifier()
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

        self.btn_add_labels = QPushButton('Draw Labels')
        self.btn_add_labels.setToolTip(
            'Add a new labels layer for manual annotations.'
        )
        self.btn_add_labels.clicked.connect(self.add_labels_layer)
        self.layout().addWidget(self.btn_add_labels)

        # 3. Training & Classification
        self.btn_train = QPushButton('Train Object Classifier')
        self.btn_train.clicked.connect(self.train_classifier)
        self.btn_train.setDisabled(True)
        self.btn_train.setToolTip(
            "Train the classifier using the 'Object Labels' and features from the selected image."
        )
        self.layout().addWidget(self.btn_train)

        self.btn_predict = QPushButton('Apply Random Forest')
        self.btn_predict.clicked.connect(self.predict_objects)
        self.btn_predict.setDisabled(True)
        self.layout().addWidget(self.btn_predict)

        # 4. IO
        self.btn_load_model = QPushButton('Load Classifier')
        self.btn_load_model.clicked.connect(self.load_model)
        self.layout().addWidget(self.btn_load_model)

        self.btn_save_model = QPushButton('Save Classifier')
        self.btn_save_model.clicked.connect(self.save_model)
        self.btn_save_model.setDisabled(True)
        self.layout().addWidget(self.btn_save_model)

        self.btn_save_labels = QPushButton('Save Labels')
        self.btn_save_labels.setToolTip(
            'Save the manually drawn labels as a TIFF file.'
        )
        self.btn_save_labels.clicked.connect(self.save_labels)
        self.layout().addWidget(self.btn_save_labels)

        # 2D Save button
        self.btn_save_preds = QPushButton('Save Predictions')
        self.btn_save_preds.setToolTip(
            'Save predicted class labels and probability maps.'
        )
        self.btn_save_preds.clicked.connect(self.save_predictions)
        self.layout().addWidget(self.btn_save_preds)

        # 3D Specific Save buttons
        self.btn_save_training_preds = QPushButton('Save Training Predictions')
        self.btn_save_training_preds.clicked.connect(
            self.save_training_predictions
        )
        self.layout().addWidget(self.btn_save_training_preds)

        self.btn_save_full_preds = QPushButton('Save Full Stack Predictions')
        self.btn_save_full_preds.clicked.connect(self.save_predictions)
        self.layout().addWidget(self.btn_save_full_preds)

        # 5. Reset
        self.btn_reset = QPushButton('Reset All')
        self.btn_reset.setToolTip(
            'Reset internal model, features, and caches to original state.'
        )
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
            'training_features': None,  # DataFrame for training
            'prediction_features': None,  # DataFrame for last processed slice
            'training_probabilities': None,  # Buffer for training slice probs (Z, C, Y, X)
            'prediction_probabilities': None,  # Buffer for full stack probs (Z, C, Y, X)
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
            and '_objects' not in layer.name
            and layer.name != 'Object Labels'
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
                if any(
                    state[x] is not None
                    for x in [
                        'training_features',
                        'prediction_features',
                        'training_probabilities',
                        'prediction_probabilities',
                    ]
                ):
                    reply = QMessageBox.question(
                        self,
                        'Clear Caches?',
                        f"You are switching to a new image. Do you want to delete the feature and probability caches for '{state['name']}' to save RAM?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes,
                    )
                    if reply == QMessageBox.Yes:
                        del self.image_states[self._current_image]
                        print(
                            f'[Object RF] Cleared state for: {state["name"]}'
                        )

            self._current_image = active_layer
            self._init_image_state(active_layer)

        # 3. Dynamic UI Renaming and Visibility
        state = self.image_states.get(active_layer)
        is_3d = state is not None and state['orig_ndim'] == 3

        if is_3d:
            self.btn_predict.setText('Apply RF to All Slices')
            self.btn_save_preds.setVisible(False)
            self.btn_save_training_preds.setVisible(True)
            self.btn_save_full_preds.setVisible(True)
        else:
            self.btn_predict.setText('Apply Random Forest')
            self.btn_save_preds.setVisible(True)
            self.btn_save_training_preds.setVisible(False)
            self.btn_save_full_preds.setVisible(False)

        # 4. Enable/Disable UI
        has_layers = active_layer is not None
        self.btn_segment.setEnabled(has_layers)
        self.btn_add_labels.setEnabled(has_layers)

        has_objects = state is not None and state['objects'] is not None
        has_manual_labels = 'Object Labels' in self.viewer.layers

        self.btn_train.setEnabled(has_objects and has_manual_labels)
        self.btn_predict.setEnabled(self._clf_ready and has_objects)
        self.btn_save_labels.setEnabled(has_manual_labels)

        has_full_preds = (
            state is not None and state['prediction_probabilities'] is not None
        )
        has_train_preds = (
            state is not None and state['training_probabilities'] is not None
        )

        self.btn_save_preds.setEnabled(has_full_preds)
        self.btn_save_training_preds.setEnabled(has_train_preds)
        self.btn_save_full_preds.setEnabled(has_full_preds)

    def add_labels_layer(self):
        """Add a new labels layer named 'Object Labels' with the correct shape."""
        active_layer = self.get_selected_layer()
        if active_layer is None:
            return

        if 'Object Labels' in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[
                'Object Labels'
            ]
            return

        # Determine shape from intensity image or objects
        state = self.image_states.get(active_layer)
        if state and state['objects'] is not None:
            shape = state['objects'].shape
        else:
            # Fallback to active layer shape
            shape = active_layer.data.shape
            if state and state['layer_type'] == 'probabilities':
                # Probabilities are (C, Y, X) or (Z, C, Y, X)
                if state['orig_ndim'] == 3:
                    # (Z, C, Y, X) -> (Z, Y, X)
                    shape = (shape[0], shape[2], shape[3])
                else:
                    # (C, Y, X) -> (Y, X)
                    shape = (shape[1], shape[2])

        self.viewer.add_labels(
            np.zeros(shape, dtype=np.uint8), name='Object Labels'
        )
        self._on_layer_change()

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

    def _get_class_map(self, prob_map):
        """Derive integer class map from probability map using argmax.
        Ensures background pixels (where all probabilities are 0) remain 0.
        """
        if prob_map is None:
            return None

        # Use classes from current fitted model
        if hasattr(self.clf, 'clf') and hasattr(self.clf.clf, 'classes_'):
            classes = self.clf.clf.classes_
        else:
            return None

        # prob_map: (Z, C, Y, X) for 3D, (C, Y, X) for 2D
        is_3d = prob_map.ndim == 4
        argmax_axis = 1 if is_3d else 0

        max_probs = np.max(prob_map, axis=argmax_axis)
        class_indices = np.argmax(prob_map, axis=argmax_axis)

        # Map indices to actual class values
        class_map = classes[class_indices].astype(np.uint8)

        # Zero out background (where all probabilities were 0)
        class_map[max_probs == 0] = 0

        return class_map

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
                return raw_labels

            areas = np.array([p.area for p in props]).reshape(-1, 1)

            # 4. Automated Thresholding (K-Means + SVM)
            # Only perform auto-thresholding if there are multiple objects
            # AND at least one object is small (potential false positive from napari-rf)
            if len(props) > 1 and np.any(areas <= 10):
                log_areas = np.log10(areas)
                kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
                cluster_labels = kmeans.fit_predict(log_areas)

                # Use SVM to find optimial separation between noise and objects
                svm = SVC(kernel='linear', C=1.0)
                svm.fit(log_areas, cluster_labels)

                # Calculate optimal threshold
                w = svm.coef_[0]
                b = svm.intercept_[0]
                log_threshold = -b / w
                threshold = 10**log_threshold

                # Ensure the threshold makes sense
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
                # Keep all objects if they are all > 10 pixels or only one object exists
                keep_labels = [p.label for p in props]

            mask = np.isin(raw_labels, keep_labels)
            filtered_labels = np.where(mask, raw_labels, 0)

            # 5. Dilation
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

            return final_labels

        def _on_done(result):
            state['objects'] = result

            new_name = f'{state["name"]}_objects'
            if new_name in self.viewer.layers:
                self.viewer.layers[new_name].data = result
            else:
                self.viewer.add_labels(result, name=new_name)

            self.btn_segment.setEnabled(True)
            self._on_layer_change()

            n_objects = len(np.unique(result)) - 1  # Exclude background 0
            print(
                f'[Object RF] Object segmentation complete. Found {n_objects} objects.'
            )

        worker = _segment_worker()
        worker.returned.connect(_on_done)
        worker.start()

    def create_features(
        self, callback=None, slice_indices=None, feature_type='prediction'
    ):
        """Extract features explicitly based on user requested slices."""
        active_image = self.get_selected_layer()
        if active_image is None or active_image not in self.image_states:
            return

        state = self.image_states[active_image]
        if state['objects'] is None:
            return

        intensity_layer = self.get_intensity_layer()
        if intensity_layer is None:
            print('[Object RF] No intensity image selected.')
            return

        self.btn_train.setEnabled(False)
        self.btn_predict.setEnabled(False)

        pbar = progress(desc='Generating Features')

        @thread_worker
        def _create_features_worker():
            yield from self.feature_extractor.generate_features(
                state['objects'],
                intensity_image=intensity_layer.data,
                indices=slice_indices,
            )

        def _on_yielded(val):
            if isinstance(val, tuple):
                step, total, desc = val
                pbar.total = total
                pbar.n = step
                pbar.set_description(desc)
                pbar.refresh()
            else:
                # Update state caches explicitly based on purpose
                if feature_type == 'training':
                    state['training_features'] = (
                        val  # TODO: check if multi-slice features are cached for training
                    )
                else:
                    state['prediction_features'] = val

        def _on_finished():
            pbar.close()
            if callback:
                callback()

        worker = _create_features_worker()
        worker.yielded.connect(_on_yielded)
        worker.finished.connect(_on_finished)
        worker.start()

    def train_classifier(self):
        """Extract features ONLY for user-labeled slices, train RF, and display training probabilities."""
        active_image = self.get_selected_layer()
        state = self.image_states.get(active_image)

        if state is None or state['objects'] is None:
            return

        if 'Object Labels' not in self.viewer.layers:
            print(
                "[Object RF] Please create a 'Object Labels' layer and annotate some objects."
            )
            return

        intensity_layer = self.get_intensity_layer()
        if intensity_layer is None:
            print('[Object RF] No intensity image selected.')
            return

        training_labels = self.viewer.layers['Object Labels'].data
        is_3d = state['orig_ndim'] == 3

        # Shape Validation Check
        if training_labels.shape != state['objects'].shape:
            print(
                f"[Object RF] Shape mismatch! 'Object Labels' shape {training_labels.shape} "
                f'does not match segmented objects shape {state["objects"].shape}. '
                "Please delete the 'Object Labels' layer and click 'Draw Labels' again."
            )
            return

        # Identify which slices the user actually annotated
        if is_3d:
            labeled_slices = np.where(
                np.any(training_labels > 0, axis=(1, 2))
            )[0].tolist()
        else:
            labeled_slices = [0] if np.any(training_labels > 0) else []

        if not labeled_slices:
            print(
                "[Object RF] No annotations found in the 'Object Labels' layer."
            )
            return

        state['labeled_slices'] = labeled_slices
        print(f'[Object RF] Detected annotations on slices: {labeled_slices}')

        self.btn_train.setEnabled(False)
        self.btn_predict.setEnabled(False)
        self.btn_segment.setEnabled(False)

        pbar = progress(desc='Extracting Training Features')

        @thread_worker
        def _train_worker():
            # 1. Generate features ONLY for annotated slices
            gen = self.feature_extractor.generate_features(
                state['objects'],
                intensity_image=intensity_layer.data,
                indices=labeled_slices,
            )

            feats_df = None
            for val in gen:
                if isinstance(val, tuple):
                    yield val
                else:
                    feats_df = val

            if feats_df is None or feats_df.empty:
                yield None
                return

            # 2. Match objects with user annotations to build training set
            X_train = []
            y_train = []
            feature_cols = [
                c for c in feats_df.columns if c not in ('label', 'slice_id')
            ]

            for _, row in feats_df.iterrows():
                lbl = int(row['label'])
                z = int(row['slice_id'])

                # Get the user's annotation within this specific object's mask
                if is_3d:
                    obj_mask = state['objects'][z] == lbl
                    ann = training_labels[z][obj_mask]
                else:
                    obj_mask = state['objects'] == lbl
                    ann = training_labels[obj_mask]

                # Use the max class ID drawn inside the object
                max_cls = np.max(ann)
                if max_cls > 0:
                    X_train.append(row[feature_cols].values)
                    y_train.append(max_cls)

            if not X_train:
                print(
                    '[Object RF] No overlapping annotations found for the objects.'
                )
                yield None
                return

            # 3. Train Classifier
            self.clf.train(X_train, y_train)

            # 4. Predict probabilities on training slices to generate a preview
            X_all = feats_df[feature_cols].values
            probas = self.clf.predict_proba(X_all)
            classes = self.clf.clf.classes_
            n_classes = len(classes)

            # Probability buffer shape: (Z, C, Y, X) for 3D, (C, Y, X) for 2D
            if is_3d:
                prob_buffer = np.zeros(
                    (
                        state['objects'].shape[0],
                        n_classes,
                        *state['objects'].shape[1:],
                    ),
                    dtype=np.float32,
                )
            else:
                prob_buffer = np.zeros(
                    (n_classes, *state['objects'].shape), dtype=np.float32
                )

            for i, (_, row) in enumerate(feats_df.iterrows()):
                lbl = int(row['label'])
                z = int(row['slice_id'])
                cls_probas = probas[i]

                if is_3d:
                    mask = state['objects'][z] == lbl
                    for c_idx in range(n_classes):
                        prob_buffer[z, c_idx][mask] = cls_probas[c_idx]
                else:
                    mask = state['objects'] == lbl
                    for c_idx in range(n_classes):
                        prob_buffer[c_idx][mask] = cls_probas[c_idx]

            yield prob_buffer

        def _on_yielded(val):
            if isinstance(val, tuple):
                step, total, desc = val
                pbar.total = total
                pbar.n = step
                pbar.set_description(desc)
                pbar.refresh()
            elif val is not None:
                # Flag ready so saving logic can proceed
                self._clf_ready = True
                state['training_probabilities'] = val

                layer_name = 'Object Training Probabilities'
                if layer_name in self.viewer.layers:
                    self.viewer.layers[layer_name].data = val
                else:
                    self.viewer.add_image(val, name=layer_name)

        def _on_finished():
            pbar.close()
            self.btn_train.setEnabled(True)
            self.btn_segment.setEnabled(True)
            if self._clf_ready:
                self.btn_predict.setEnabled(True)
                self.btn_save_model.setEnabled(True)
                print('[Object RF] Training completed successfully.')

        worker = _train_worker()
        worker.yielded.connect(_on_yielded)
        worker.finished.connect(_on_finished)
        worker.start()

    def predict_objects(self):
        """Apply RF to all slices. Extracts features -> predicts -> discards features to save memory."""
        active_image = self.get_selected_layer()
        if active_image is None or not self._clf_ready:
            return

        state = self.image_states.get(active_image)
        intensity_layer = self.get_intensity_layer()
        is_3d = state['orig_ndim'] == 3

        if not intensity_layer:
            print('[Object RF] No intensity image selected.')
            return

        self.btn_train.setEnabled(False)
        self.btn_predict.setEnabled(False)
        self.btn_segment.setEnabled(False)

        print(f'[Object RF] Applying Random Forest to full {state["name"]}...')
        pbar = progress(desc='Applying Random Forest')

        @thread_worker
        def _apply_rf_worker():
            classes = self.clf.clf.classes_
            n_classes = len(classes)

            if is_3d:
                prob_results = np.zeros(
                    (
                        state['objects'].shape[0],
                        n_classes,
                        *state['objects'].shape[1:],
                    ),
                    dtype=np.float32,
                )
            else:
                prob_results = np.zeros(
                    (n_classes, *state['objects'].shape), dtype=np.float32
                )

            if is_3d:
                total_slices = state['objects'].shape[0]
                for z in range(total_slices):
                    # Hybrid Workflow: Reuse training features if available, otherwise generate
                    if (
                        z in state['labeled_slices']
                        and state['training_features'] is not None
                    ):
                        # Find the rows corresponding to this slice
                        tf = state['training_features']
                        feats_df = tf[tf['slice_id'] == z].copy()
                        yield (
                            z + 1,
                            total_slices,
                            f'Slice {z + 1}/{total_slices}: Reusing training features',
                        )
                    else:
                        gen = self.feature_extractor.generate_features(
                            state['objects'],
                            intensity_image=intensity_layer.data,
                            indices=[z],
                        )
                        feats_df = None
                        for val in gen:
                            if isinstance(val, tuple):
                                yield (
                                    z + 1,
                                    total_slices,
                                    f'Slice {z + 1}/{total_slices}: {val[2]}',
                                )
                            else:
                                feats_df = val

                    if feats_df is not None and not feats_df.empty:
                        feature_cols = [
                            c
                            for c in feats_df.columns
                            if c not in ('label', 'slice_id')
                        ]
                        X = feats_df[feature_cols].values
                        probas = self.clf.predict_proba(X)

                        slice_objs = state['objects'][z]
                        for i, (_, row) in enumerate(feats_df.iterrows()):
                            lbl = int(row['label'])
                            mask = slice_objs == lbl
                            for c_idx in range(n_classes):
                                prob_results[z, c_idx][mask] = probas[i][c_idx]

                    if z == total_slices - 1:
                        state['prediction_features'] = feats_df
                yield prob_results
            else:
                # 2D: Use existing cache if valid
                if state['prediction_features'] is None:
                    gen = self.feature_extractor.generate_features(
                        state['objects'], intensity_image=intensity_layer.data
                    )
                    feats_df = None
                    for val in gen:
                        if isinstance(val, tuple):
                            yield (0, 1, val[2])
                        else:
                            feats_df = val
                    state['prediction_features'] = feats_df

                feats_df = state['prediction_features']
                if feats_df is not None and not feats_df.empty:
                    feature_cols = [
                        c
                        for c in feats_df.columns
                        if c not in ('label', 'slice_id')
                    ]
                    X = feats_df[feature_cols].values
                    probas = self.clf.predict_proba(X)

                    slice_objs = state['objects']
                    for i, (_, row) in enumerate(feats_df.iterrows()):
                        lbl = int(row['label'])
                        mask = slice_objs == lbl
                        for c_idx in range(n_classes):
                            prob_results[c_idx][mask] = probas[i][c_idx]
                yield prob_results

        def _on_yielded(val):
            if isinstance(val, tuple):
                z, total, desc = val
                pbar.total, pbar.n = total, z
                pbar.set_description(desc)
                pbar.refresh()
            else:
                state['prediction_probabilities'] = val

                layer_name = 'Object Probabilities'
                if layer_name in self.viewer.layers:
                    self.viewer.layers[layer_name].data = val
                else:
                    self.viewer.add_image(val, name=layer_name)

        def _on_finished():
            pbar.close()
            self.btn_predict.setEnabled(True)
            self.btn_train.setEnabled(True)
            self.btn_segment.setEnabled(True)
            self._on_layer_change()
            print('[Object RF] Random Forest application complete.')

        worker = _apply_rf_worker()
        worker.yielded.connect(_on_yielded)
        worker.finished.connect(_on_finished)
        worker.start()

    def _get_save_dir(self, state):
        intensity_layer = self.get_intensity_layer()
        if intensity_layer is None:
            return Path.home() / 'object_rf_export'

        # Try to get path from source
        path = None
        source = getattr(intensity_layer, 'source', None)
        raw_path = getattr(source, 'path', None)
        if raw_path:
            path = str(
                raw_path[0]
                if isinstance(raw_path, (list, tuple))
                else raw_path
            )

        if path:
            save_dir = Path(path).parent / intensity_layer.name
        else:
            save_dir = Path.home() / intensity_layer.name

        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def save_labels(self):
        if 'Object Labels' in self.viewer.layers:
            try:
                state = self.image_states.get(self._current_image)
                labels = self.viewer.layers['Object Labels'].data

                is_3d = state and state['orig_ndim'] == 3
                if state and not is_3d and labels.ndim == 3:
                    labels = np.max(labels, axis=0)
                elif state and is_3d and labels.ndim == 4:
                    labels = np.max(labels, axis=1)

                save_dir = self._get_save_dir(state)
                intensity_layer = self.get_intensity_layer()
                name = intensity_layer.name if intensity_layer else 'labels'
                save_path = save_dir / f'{name}_object_manual_labels.tif'
                imsave(
                    str(save_path),
                    labels.astype(np.uint8),
                    check_contrast=False,
                )
                print(f'[Object RF] Saved labels to {save_path}')
            except (OSError, ValueError, RuntimeError) as e:
                print(f'[Object RF] Error in save_labels: {e}')
        else:
            print("[Object RF] No 'Object Labels' layer found to save.")

    def save_predictions(self):
        active_image = self.get_selected_layer()
        state = self.image_states.get(active_image)
        if state and state['prediction_probabilities'] is not None:
            self._save_state_outputs(
                state, 'prediction_probabilities', 'object_full'
            )

    def save_training_predictions(self):
        active_image = self.get_selected_layer()
        state = self.image_states.get(active_image)
        if state and state['training_probabilities'] is not None:
            self._save_state_outputs(
                state, 'training_probabilities', 'object_training'
            )

    def _save_state_outputs(self, state, prob_key, suffix):
        prob_map = state[prob_key]
        class_map = self._get_class_map(prob_map)

        save_dir = self._get_save_dir(state)
        intensity_layer = self.get_intensity_layer()
        name = intensity_layer.name if intensity_layer else state['name']

        # Save Integer Class Labels
        imsave(
            str(save_dir / f'{name}_{suffix}_class.tif'),
            class_map.astype(np.uint8),
            check_contrast=False,
        )

        # Save Float Probability Maps
        imsave(
            str(save_dir / f'{name}_{suffix}_probs.tif'),
            prob_map.astype(np.float32),
            check_contrast=False,
        )

        print(
            f'[Object RF] Saved {suffix} class map and probabilities to {save_dir}'
        )

    def save_model(self):
        if self.clf is None:
            return
        state = self.image_states.get(self._current_image)
        save_dir = self._get_save_dir(state)
        default_path = str(save_dir / 'object_classifier.joblib')

        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Model', default_path, 'Model (*.joblib)'
        )
        if save_path:
            dump(self.clf.clf, save_path)  # Save the internal sklearn model
            print(f'Model saved to {save_path}')

    def load_model(self):
        load_path, _ = QFileDialog.getOpenFileName(
            self, 'Load Model', '', 'Model (*.joblib)'
        )
        if load_path:
            self.clf = ObjectClassifier()
            self.clf.clf = load(load_path)  # Load directly into the wrapper
            self._clf_ready = True
            self._on_layer_change()
            print(f'Model loaded from {load_path}')

    def reset_all(self):
        """Reset internal state to default."""
        self.clf = ObjectClassifier()
        self.image_states.clear()
        self._current_image = None
        self._clf_ready = False
        self._on_layer_change()
