from qtpy.QtCore import Signal
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget


class ClassifierControlsWidget(QWidget):
    train_requested = Signal()
    predict_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())

        # 3. Training & Classification
        self.btn_train = QPushButton('Train Object Classifier')
        self.btn_train.clicked.connect(self.train_requested)
        self.btn_train.setDisabled(True)
        self.btn_train.setToolTip(
            "Train the classifier using the 'Object Labels' and features from the selected image."
        )
        self.layout().addWidget(self.btn_train)

        self.btn_predict = QPushButton('Apply Random Forest')
        self.btn_predict.clicked.connect(self.predict_requested)
        self.btn_predict.setDisabled(True)
        self.layout().addWidget(self.btn_predict)

    def set_3d_mode(self, is_3d: bool):
        if is_3d:
            self.btn_predict.setText('Apply RF to All Slices')
        else:
            self.btn_predict.setText('Apply Random Forest')

    def set_training_enabled(self, enabled: bool):
        self.btn_train.setEnabled(enabled)

    def set_predict_enabled(self, enabled: bool):
        self.btn_predict.setEnabled(enabled)
