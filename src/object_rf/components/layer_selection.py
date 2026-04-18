from typing import TYPE_CHECKING

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

if TYPE_CHECKING:
    import napari


class LayerSelectionWidget(QWidget):
    layer_changed = Signal()

    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())

        # 1. Image Layer Selection (Probabilities or masks)
        self.layout().addWidget(QLabel('Select Probability/Label Layer:'))
        self.layer_combo = QComboBox()
        self.layer_combo.setToolTip(
            'Select the image (probabilities) or label layer to segment.'
        )
        self.layer_combo.currentIndexChanged.connect(self.layer_changed)
        self.layout().addWidget(self.layer_combo)

        # 2. Original Image Selection (Intensity Source)
        self.layout().addWidget(QLabel('Select Original Intensity Image:'))
        self.image_combo = QComboBox()
        self.image_combo.setToolTip(
            'Select the original image to use as the intensity source for features.'
        )
        self.layout().addWidget(self.image_combo)

    def update_layers(self, viewer: 'napari.Viewer'):
        import napari

        current_layer_name = self.layer_combo.currentText()
        current_image_name = self.image_combo.currentText()

        self.layer_combo.blockSignals(True)
        self.image_combo.blockSignals(True)

        self.layer_combo.clear()
        self.image_combo.clear()

        image_layers = [
            layer
            for layer in viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]
        label_layers = [
            layer
            for layer in viewer.layers
            if isinstance(layer, napari.layers.Labels)
            and '_objects' not in layer.name
            and layer.name != 'Object Labels'
        ]

        all_candidate_layers = image_layers + label_layers
        self.layer_combo.addItems(
            [layer.name for layer in all_candidate_layers]
        )
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

    def get_selected_layer(self, viewer: 'napari.Viewer'):
        name = self.layer_combo.currentText()
        if name in viewer.layers:
            return viewer.layers[name]
        return None

    def get_intensity_layer(self, viewer: 'napari.Viewer'):
        name = self.image_combo.currentText()
        if name in viewer.layers:
            return viewer.layers[name]
        return None
