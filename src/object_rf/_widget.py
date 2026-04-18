from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import dump, load
from napari.qt.threading import thread_worker
from napari.utils import progress
from qtpy.QtWidgets import QFileDialog, QMessageBox, QVBoxLayout, QWidget
from skimage.io import imsave

from .classifier import ObjectClassifier
from .components.action_buttons import ActionButtonsWidget
from .components.classifier_controls import ClassifierControlsWidget
from .components.io_controls import IOControlsWidget
from .components.layer_selection import LayerSelectionWidget
from .feature_extraction import FeatureExtractor
from .state import ImageStateManager
from .utils import get_class_map, get_save_directory
from .workers import (
    apply_rf_worker,
    segment_objects_worker,
    train_classifier_worker,
)

if TYPE_CHECKING:
    import napari


class ObjectWidget(QWidget):
    def __init__(self, viewer: 'napari.viewer.Viewer'):
        super().__init__()
        self.viewer = viewer

        # State management
        self.state_manager = ImageStateManager()
        self.clf = ObjectClassifier()
        self._clf_ready = False
        self.feature_extractor = FeatureExtractor()

        # --- UI Components ---
        self.setLayout(QVBoxLayout())

        self.layer_selection = LayerSelectionWidget()
        self.layer_selection.layer_changed.connect(self._on_layer_change)
        self.layout().addWidget(self.layer_selection)

        self.action_buttons = ActionButtonsWidget()
        self.action_buttons.segment_requested.connect(self.segment_objects)
        self.action_buttons.add_labels_requested.connect(self.add_labels_layer)
        self.layout().addWidget(self.action_buttons)

        self.classifier_controls = ClassifierControlsWidget()
        self.classifier_controls.train_requested.connect(self.train_classifier)
        self.classifier_controls.predict_requested.connect(
            self.predict_objects
        )
        self.layout().addWidget(self.classifier_controls)

        self.io_controls = IOControlsWidget()
        self.io_controls.load_model_requested.connect(self.load_model)
        self.io_controls.save_model_requested.connect(self.save_model)
        self.io_controls.save_labels_requested.connect(self.save_labels)
        self.io_controls.save_predictions_requested.connect(
            self.save_predictions
        )
        self.io_controls.save_training_predictions_requested.connect(
            self.save_training_predictions
        )
        self.io_controls.save_features_requested.connect(self.save_features)
        self.io_controls.reset_requested.connect(self.reset_all)
        self.layout().addWidget(self.io_controls)

        # Connect layer events to keep the dropdown updated
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)
        self._on_layer_change()

    def _on_layer_change(self, event=None):
        """Update dropdowns and manage state transition between image layers."""
        import napari
        from qtpy.QtCore import QTimer

        # Handle automatic conversion of Image layers to Labels if they look like annotations
        # We use QTimer.singleShot(0, ...) to defer the replacement until the next event loop cycle,
        # preventing ValueError when napari tries to select a layer we just removed during insertion.
        for layer in list(self.viewer.layers):
            if isinstance(layer, napari.layers.Image) and layer.name.endswith(
                '_object_manual_labels'
            ):

                def replace_layer(layer_to_replace=layer):
                    if layer_to_replace in self.viewer.layers:
                        data = layer_to_replace.data.astype(np.uint8)
                        name = layer_to_replace.name
                        self.viewer.layers.remove(layer_to_replace)
                        new_layer = self.viewer.add_labels(data, name=name)
                        self.viewer.layers.selection.active = new_layer

                QTimer.singleShot(0, replace_layer)
                return

        # 1. Update Dropdowns
        self.layer_selection.update_layers(self.viewer)

        # 2. Handle State Migration
        active_layer = self.layer_selection.get_selected_layer(self.viewer)
        if active_layer != self.state_manager.current_image:
            if (
                self.state_manager.current_image
                and self.state_manager.current_image
                in self.state_manager.image_states
            ):
                state = self.state_manager.get_state(
                    self.state_manager.current_image
                )
                if any(
                    state[x] is not None
                    for x in [
                        'full_feature_table',
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
                        self.state_manager.clear_state(
                            self.state_manager.current_image
                        )
                        print(
                            f'[Object RF] Cleared state for: {state["name"]}'
                        )

            self.state_manager.current_image = active_layer
            self.state_manager.init_image_state(active_layer)

        # 3. Dynamic UI Updates
        state = self.state_manager.get_state(active_layer)
        is_3d = state is not None and state['orig_ndim'] == 3

        self.classifier_controls.set_3d_mode(is_3d)
        self.io_controls.set_3d_mode(is_3d)

        # 4. Enable/Disable UI
        has_layers = active_layer is not None
        self.action_buttons.set_enabled(has_layers)

        has_objects = state is not None and state['objects'] is not None
        manual_labels_layer = self.get_manual_labels_layer()
        has_manual_labels = manual_labels_layer is not None

        self.classifier_controls.set_training_enabled(
            has_objects and has_manual_labels
        )
        self.classifier_controls.set_predict_enabled(
            self._clf_ready and has_objects
        )

        has_full_preds = (
            state is not None and state['prediction_probabilities'] is not None
        )
        has_train_preds = (
            state is not None and state['training_probabilities'] is not None
        )
        has_features = (
            state is not None and state['full_feature_table'] is not None
        )

        self.io_controls.set_save_model_enabled(self._clf_ready)
        self.io_controls.set_save_labels_enabled(has_manual_labels)
        self.io_controls.set_save_preds_enabled(has_full_preds)
        self.io_controls.set_save_training_preds_enabled(has_train_preds)
        self.io_controls.set_save_features_enabled(has_features)

    def add_labels_layer(self):
        """Add a new labels layer named 'Object Labels' with the correct shape."""
        active_layer = self.layer_selection.get_selected_layer(self.viewer)
        if active_layer is None:
            return

        manual_labels_layer = self.get_manual_labels_layer()
        if manual_labels_layer is not None:
            self.viewer.layers.selection.active = manual_labels_layer
            return

        state = self.state_manager.get_state(active_layer)
        if state and state['objects'] is not None:
            shape = state['objects'].shape
        else:
            shape = active_layer.data.shape
            if state and state['layer_type'] == 'probabilities':
                if state['orig_ndim'] == 3:
                    shape = (shape[0], shape[2], shape[3])
                else:
                    shape = (shape[1], shape[2])

        self.viewer.add_labels(
            np.zeros(shape, dtype=np.uint8), name='Object Labels'
        )
        self._on_layer_change()

    def get_manual_labels_layer(self):
        """Find the manual labels layer."""
        import napari

        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels) and (
                layer.name == 'Object Labels'
                or layer.name.endswith('_object_manual_labels')
            ):
                return layer
        return None

    def segment_objects(self):
        layer = self.layer_selection.get_selected_layer(self.viewer)
        if layer is None:
            return

        self.state_manager.init_image_state(layer)
        state = self.state_manager.get_state(layer)
        if state is None:
            return

        self.action_buttons.btn_segment.setEnabled(False)

        @thread_worker
        def _worker():
            return segment_objects_worker(
                layer.data, state['orig_ndim'], state['layer_type']
            )

        def _on_done(result):
            state['objects'] = result
            new_name = f'{state["name"]}_objects'
            if new_name in self.viewer.layers:
                self.viewer.layers[new_name].data = result
            else:
                self.viewer.add_labels(result, name=new_name)

            self.action_buttons.btn_segment.setEnabled(True)
            self._on_layer_change()

            n_objects = len(np.unique(result)) - 1
            print(
                f'[Object RF] Object segmentation complete. Found {n_objects} objects.'
            )

        worker = _worker()
        worker.returned.connect(_on_done)
        worker.start()

    def train_classifier(self):
        active_image = self.layer_selection.get_selected_layer(self.viewer)
        state = self.state_manager.get_state(active_image)

        if state is None or state['objects'] is None:
            return

        manual_labels_layer = self.get_manual_labels_layer()
        intensity_layer = self.layer_selection.get_intensity_layer(self.viewer)

        if not manual_labels_layer or not intensity_layer:
            return

        training_labels = manual_labels_layer.data
        if training_labels.shape != state['objects'].shape:
            print('[Object RF] Shape mismatch between labels and objects!')
            return

        is_3d = state['orig_ndim'] == 3
        if is_3d:
            labeled_slices = np.where(
                np.any(training_labels > 0, axis=(1, 2))
            )[0].tolist()
        else:
            labeled_slices = [0] if np.any(training_labels > 0) else []

        if not labeled_slices:
            return

        state['labeled_slices'] = labeled_slices
        self.classifier_controls.btn_train.setEnabled(False)
        self.classifier_controls.btn_predict.setEnabled(False)
        self.action_buttons.btn_segment.setEnabled(False)

        pbar = progress(desc='Extracting Training Features')

        @thread_worker
        def _worker():
            yield from train_classifier_worker(
                state['objects'],
                intensity_layer.data,
                training_labels,
                labeled_slices,
                state['orig_ndim'],
                self.feature_extractor,
                self.clf,
            )

        def _on_yielded(val):
            if isinstance(val, tuple):
                step, total, desc = val
                pbar.total, pbar.n = total, step
                pbar.set_description(desc)
                pbar.refresh()
            elif val is not None:
                feats_df, prob_buffer = val
                self._clf_ready = True
                state['training_probabilities'] = prob_buffer

                # Update state feature table
                if state['full_feature_table'] is None:
                    state['full_feature_table'] = feats_df
                else:
                    existing_df = state['full_feature_table']
                    if 'true_label' not in existing_df.columns:
                        existing_df['true_label'] = 0
                    existing_df = existing_df[
                        ~existing_df['slice_id'].isin(labeled_slices)
                    ]
                    state['full_feature_table'] = pd.concat(
                        [existing_df, feats_df], ignore_index=True
                    )

                layer_name = 'Object Training Probabilities'
                if layer_name in self.viewer.layers:
                    self.viewer.layers[layer_name].data = prob_buffer
                else:
                    self.viewer.add_image(prob_buffer, name=layer_name)

        def _on_finished():
            pbar.close()
            self.classifier_controls.btn_train.setEnabled(True)
            self.action_buttons.btn_segment.setEnabled(True)
            if self._clf_ready:
                self.classifier_controls.btn_predict.setEnabled(True)
                self.io_controls.btn_save_model.setEnabled(True)
                self._on_layer_change()
                print('[Object RF] Training completed successfully.')

        worker = _worker()
        worker.yielded.connect(_on_yielded)
        worker.finished.connect(_on_finished)
        worker.start()

    def predict_objects(self):
        active_image = self.layer_selection.get_selected_layer(self.viewer)
        if active_image is None or not self._clf_ready:
            return

        state = self.state_manager.get_state(active_image)
        intensity_layer = self.layer_selection.get_intensity_layer(self.viewer)
        if not intensity_layer:
            return

        self.classifier_controls.btn_train.setEnabled(False)
        self.classifier_controls.btn_predict.setEnabled(False)
        self.action_buttons.btn_segment.setEnabled(False)

        pbar = progress(desc='Applying Random Forest')

        @thread_worker
        def _worker():
            yield from apply_rf_worker(
                state['objects'],
                intensity_layer.data,
                state['orig_ndim'],
                state['full_feature_table'],
                self.feature_extractor,
                self.clf,
            )

        def _on_yielded(val):
            if isinstance(val, tuple) and len(val) == 3:
                z, total, desc = val
                pbar.total, pbar.n = total, z
                pbar.set_description(desc)
                pbar.refresh()
            elif val is not None:
                final_table, prob_results = val
                if final_table is not None:
                    state['full_feature_table'] = final_table
                state['prediction_probabilities'] = prob_results

                layer_name = 'Object Probabilities'
                if layer_name in self.viewer.layers:
                    self.viewer.layers[layer_name].data = prob_results
                else:
                    self.viewer.add_image(prob_results, name=layer_name)

        def _on_finished():
            pbar.close()
            self.classifier_controls.btn_predict.setEnabled(True)
            self.classifier_controls.btn_train.setEnabled(True)
            self.action_buttons.btn_segment.setEnabled(True)
            self._on_layer_change()
            print('[Object RF] Random Forest application complete.')

        worker = _worker()
        worker.yielded.connect(_on_yielded)
        worker.finished.connect(_on_finished)
        worker.start()

    def save_labels(self):
        manual_labels_layer = self.get_manual_labels_layer()
        if manual_labels_layer is not None:
            state = self.state_manager.get_state(
                self.state_manager.current_image
            )
            labels = manual_labels_layer.data
            is_3d = state and state['orig_ndim'] == 3
            if state and not is_3d and labels.ndim == 3:
                labels = np.max(labels, axis=0)
            elif state and is_3d and labels.ndim == 4:
                labels = np.max(labels, axis=1)

            intensity_layer = self.layer_selection.get_intensity_layer(
                self.viewer
            )
            save_dir = get_save_directory(
                intensity_layer,
                intensity_layer.name if intensity_layer else 'labels',
            )
            name = intensity_layer.name if intensity_layer else 'labels'
            save_path = save_dir / f'{name}_object_manual_labels.tif'
            imsave(
                str(save_path), labels.astype(np.uint8), check_contrast=False
            )
            print(f'[Object RF] Saved labels to {save_path}')

    def save_predictions(self):
        state = self.state_manager.get_state(self.state_manager.current_image)
        if state and state['prediction_probabilities'] is not None:
            self._save_state_outputs(
                state, 'prediction_probabilities', 'object_full'
            )

    def save_training_predictions(self):
        state = self.state_manager.get_state(self.state_manager.current_image)
        if state and state['training_probabilities'] is not None:
            self._save_state_outputs(
                state, 'training_probabilities', 'object_training'
            )

    def _save_state_outputs(self, state, prob_key, suffix):
        prob_map = state[prob_key]
        classes = self.clf.clf.classes_
        class_map = get_class_map(prob_map, classes)

        intensity_layer = self.layer_selection.get_intensity_layer(self.viewer)
        save_dir = get_save_directory(
            intensity_layer,
            intensity_layer.name if intensity_layer else state['name'],
        )
        name = intensity_layer.name if intensity_layer else state['name']

        imsave(
            str(save_dir / f'{name}_{suffix}_class.tif'),
            class_map.astype(np.uint8),
            check_contrast=False,
        )
        imsave(
            str(save_dir / f'{name}_{suffix}_probs.tif'),
            prob_map.astype(np.float32),
            check_contrast=False,
        )
        print(
            f'[Object RF] Saved {suffix} class map and probabilities to {save_dir}'
        )

    def save_features(self):
        state = self.state_manager.get_state(self.state_manager.current_image)
        if state and state['full_feature_table'] is not None:
            intensity_layer = self.layer_selection.get_intensity_layer(
                self.viewer
            )
            save_dir = get_save_directory(
                intensity_layer,
                intensity_layer.name if intensity_layer else state['name'],
            )
            name = intensity_layer.name if intensity_layer else state['name']
            save_path = save_dir / f'{name}_full_features.csv'
            state['full_feature_table'].to_csv(save_path, index=False)
            print(f'[Object RF] Saved full feature table to {save_path}')

    def save_model(self):
        state = self.state_manager.get_state(self.state_manager.current_image)
        intensity_layer = self.layer_selection.get_intensity_layer(self.viewer)
        save_dir = get_save_directory(
            intensity_layer,
            intensity_layer.name if intensity_layer else state['name'],
        )
        default_path = str(save_dir / 'object_classifier.joblib')

        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Model', default_path, 'Model (*.joblib)'
        )
        if save_path:
            dump(self.clf.clf, save_path)
            print(f'Model saved to {save_path}')

    def load_model(self):
        load_path, _ = QFileDialog.getOpenFileName(
            self, 'Load Model', '', 'Model (*.joblib)'
        )
        if load_path:
            self.clf = ObjectClassifier()
            self.clf.clf = load(load_path)
            self._clf_ready = True
            self._on_layer_change()
            print(f'Model loaded from {load_path}')

    def reset_all(self):
        self.clf = ObjectClassifier()
        self.state_manager.reset_all()
        self._clf_ready = False
        self._on_layer_change()
