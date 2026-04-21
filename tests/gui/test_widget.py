import numpy as np

from object_xgb._widget import ObjectWidget


def test_object_widget_init(make_napari_viewer):
    # Test that the widget can be instantiated
    viewer = make_napari_viewer()
    widget = ObjectWidget(viewer)
    assert widget.viewer == viewer
    assert widget.prob_layer_combo.count() == 0


def test_object_widget_layer_sync(make_napari_viewer):
    # Test that the dropdown updates when layers are added/removed
    viewer = make_napari_viewer()
    widget = ObjectWidget(viewer)

    # Add an image layer
    viewer.add_image(np.random.random((10, 10)), name='test_image')
    assert widget.prob_layer_combo.count() == 1
    assert widget.prob_layer_combo.currentText() == 'test_image'

    # Add another layer
    viewer.add_labels(np.zeros((10, 10), dtype=int), name='test_labels')
    assert widget.prob_layer_combo.count() == 2

    # Remove a layer
    viewer.layers.remove('test_image')
    assert widget.prob_layer_combo.count() == 1
    assert widget.prob_layer_combo.currentText() == 'test_labels'


def test_segment_objects_binary(make_napari_viewer):
    # Test segmentation from a binary mask
    viewer = make_napari_viewer()
    widget = ObjectWidget(viewer)

    mask = np.zeros((10, 10), dtype=int)
    mask[2:5, 2:5] = 1
    mask[7:9, 7:9] = 1
    viewer.add_labels(mask, name='mask')

    widget.prob_layer_combo.setCurrentText('mask')
    widget.segment_objects()

    assert 'mask_objects' in viewer.layers
    objects_data = viewer.layers['mask_objects'].data
    assert np.max(objects_data) == 2  # Two objects


def test_segment_objects_probabilities(make_napari_viewer):
    # Test segmentation from a multi-channel probability map
    viewer = make_napari_viewer()
    widget = ObjectWidget(viewer)

    # 3 channels (Background, Object A, Object B), 10x10
    probs = np.zeros((3, 10, 10), dtype=float)
    probs[0, :, :] = 0.2  # Background
    probs[1, 2:5, 2:5] = 0.8  # Object A
    probs[2, 7:9, 7:9] = 0.8  # Object B

    viewer.add_image(probs, name='probs')
    widget.prob_layer_combo.setCurrentText('probs')
    widget.segment_objects()

    assert 'probs_objects' in viewer.layers
    objects_data = viewer.layers['probs_objects'].data
    # argmax will pick index 1 for the first object and index 2 for the second
    # measure.label will find 2 distinct foreground regions
    assert np.max(objects_data) == 2
