import numpy as np

from object_xgb.workers import segment_objects_worker


def test_segmentation_worker_2d_float():
    # Simulate (C, Y, X) probability map
    data = np.zeros((3, 32, 32), dtype=float)
    # Background (class 0)
    data[0, ...] = 0.8
    # Class 1 patch
    data[1, 5:15, 5:15] = 0.9
    # Class 2 patch
    data[2, 20:25, 20:25] = 0.9

    labels = segment_objects_worker(
        data, orig_ndim=2, layer_type='probabilities'
    )

    assert labels.max() >= 2  # At least two distinct patches
    assert np.any(labels[5:15, 5:15] > 0)
    assert np.any(labels[20:25, 20:25] > 0)


def test_segmentation_worker_3d_float():
    # Simulate (Z, C, Y, X) probability map
    data = np.zeros((10, 3, 32, 32), dtype=float)
    # Background
    data[:, 0, ...] = 0.8
    # Class 1 patch in slice 5
    data[5, 1, 5:15, 5:15] = 0.9

    labels = segment_objects_worker(
        data, orig_ndim=3, layer_type='probabilities'
    )

    assert labels.max() >= 1
    assert np.any(labels[5, 5:15, 5:15] > 0)


def test_segmentation_worker_2d_int():
    # Simulate (Y, X) binary mask
    data = np.zeros((32, 32), dtype=int)
    data[5:15, 5:15] = 1

    labels = segment_objects_worker(data, orig_ndim=2, layer_type='labels')

    assert labels.max() >= 1
    assert np.any(labels[5:15, 5:15] > 0)
