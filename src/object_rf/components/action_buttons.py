from qtpy.QtCore import Signal
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget


class ActionButtonsWidget(QWidget):
    segment_requested = Signal()
    add_labels_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setLayout(QVBoxLayout())

        self.btn_segment = QPushButton('Segment Objects')
        self.btn_segment.setToolTip(
            'Generate unique object labels and automatically filter noise.'
        )
        self.btn_segment.clicked.connect(self.segment_requested)
        self.layout().addWidget(self.btn_segment)

        self.btn_add_labels = QPushButton('Draw Labels')
        self.btn_add_labels.setToolTip(
            'Add a new labels layer for manual annotations.'
        )
        self.btn_add_labels.clicked.connect(self.add_labels_requested)
        self.layout().addWidget(self.btn_add_labels)

    def set_enabled(self, enabled: bool):
        self.btn_segment.setEnabled(enabled)
        self.btn_add_labels.setEnabled(enabled)
