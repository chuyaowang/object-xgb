from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ClassifierControlsWidget(QWidget):
    train_requested = Signal()
    predict_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())

        # 1. Feature Selection Threshold Slider
        self.layout().addWidget(QLabel('Feature Selection Threshold (VIP):'))
        thresh_layout = QHBoxLayout()

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(50)  # Represents 0.0 to 5.0
        self.threshold_slider.setValue(10)  # Default 1.0
        self.threshold_slider.setToolTip(
            'Threshold for pairwise PLS-DA VIP scores. Higher values = fewer features.'
        )

        self.threshold_label = QLabel('1.0')
        self.threshold_slider.valueChanged.connect(self._on_slider_changed)

        thresh_layout.addWidget(self.threshold_slider)
        thresh_layout.addWidget(self.threshold_label)
        self.layout().addLayout(thresh_layout)

        # 2. Training & Classification Buttons
        self.btn_train = QPushButton('Train Object Classifier')
        self.btn_train.clicked.connect(self.train_requested)
        self.btn_train.setDisabled(True)
        self.btn_train.setToolTip(
            "Train the classifier using the 'Object Labels' and features from the selected image."
        )
        self.layout().addWidget(self.btn_train)

        self.btn_predict = QPushButton('Apply XGBoost Pipeline')
        self.btn_predict.clicked.connect(self.predict_requested)
        self.btn_predict.setDisabled(True)
        self.layout().addWidget(self.btn_predict)

    def _on_slider_changed(self, value):
        self.threshold_label.setText(f'{value / 10:.1f}')

    def get_threshold(self) -> float:
        return self.threshold_slider.value() / 10

    def set_3d_mode(self, is_3d: bool):
        if is_3d:
            self.btn_predict.setText('Apply XGBoost to All Slices')
        else:
            self.btn_predict.setText('Apply XGBoost Pipeline')

    def set_training_enabled(self, enabled: bool):
        self.btn_train.setEnabled(enabled)

    def set_predict_enabled(self, enabled: bool):
        self.btn_predict.setEnabled(enabled)
