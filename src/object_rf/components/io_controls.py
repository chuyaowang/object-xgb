from qtpy.QtCore import Signal
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget


class IOControlsWidget(QWidget):
    load_model_requested = Signal()
    save_model_requested = Signal()
    save_labels_requested = Signal()
    save_predictions_requested = Signal()
    save_training_predictions_requested = Signal()
    save_features_requested = Signal()
    reset_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())

        # 4. IO
        self.btn_load_model = QPushButton('Load Classifier')
        self.btn_load_model.clicked.connect(self.load_model_requested)
        self.layout().addWidget(self.btn_load_model)

        self.btn_save_model = QPushButton('Save Classifier')
        self.btn_save_model.clicked.connect(self.save_model_requested)
        self.btn_save_model.setDisabled(True)
        self.layout().addWidget(self.btn_save_model)

        self.btn_save_labels = QPushButton('Save Labels')
        self.btn_save_labels.setToolTip(
            'Save the manually drawn labels as a TIFF file.'
        )
        self.btn_save_labels.clicked.connect(self.save_labels_requested)
        self.layout().addWidget(self.btn_save_labels)

        # 2D Save button
        self.btn_save_preds = QPushButton('Save Predictions')
        self.btn_save_preds.setToolTip(
            'Save predicted class labels and probability maps.'
        )
        self.btn_save_preds.clicked.connect(self.save_predictions_requested)
        self.layout().addWidget(self.btn_save_preds)

        # 3D Specific Save buttons
        self.btn_save_training_preds = QPushButton('Save Training Predictions')
        self.btn_save_training_preds.clicked.connect(
            self.save_training_predictions_requested
        )
        self.layout().addWidget(self.btn_save_training_preds)

        self.btn_save_full_preds = QPushButton('Save Full Stack Predictions')
        self.btn_save_full_preds.clicked.connect(
            self.save_predictions_requested
        )
        self.layout().addWidget(self.btn_save_full_preds)

        self.btn_save_features = QPushButton('Save Features (CSV)')
        self.btn_save_features.setToolTip(
            'Save the full feature matrix for all objects as a CSV file.'
        )
        self.btn_save_features.clicked.connect(self.save_features_requested)
        self.layout().addWidget(self.btn_save_features)

        # 5. Reset
        self.btn_reset = QPushButton('Reset All')
        self.btn_reset.setToolTip(
            'Reset internal model, features, and caches to original state.'
        )
        self.btn_reset.clicked.connect(self.reset_requested)
        self.layout().addWidget(self.btn_reset)

    def set_3d_mode(self, is_3d: bool):
        if is_3d:
            self.btn_save_preds.setVisible(False)
            self.btn_save_training_preds.setVisible(True)
            self.btn_save_full_preds.setVisible(True)
        else:
            self.btn_save_preds.setVisible(True)
            self.btn_save_training_preds.setVisible(False)
            self.btn_save_full_preds.setVisible(False)

    def set_save_model_enabled(self, enabled: bool):
        self.btn_save_model.setEnabled(enabled)

    def set_save_labels_enabled(self, enabled: bool):
        self.btn_save_labels.setEnabled(enabled)

    def set_save_preds_enabled(self, enabled: bool):
        self.btn_save_preds.setEnabled(enabled)
        self.btn_save_full_preds.setEnabled(enabled)

    def set_save_training_preds_enabled(self, enabled: bool):
        self.btn_save_training_preds.setEnabled(enabled)

    def set_save_features_enabled(self, enabled: bool):
        self.btn_save_features.setEnabled(enabled)
