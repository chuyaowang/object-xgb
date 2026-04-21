import numpy as np

from object_xgb._writer import write_multiple, write_single_image


def test_write_single_image(tmp_path):
    path = str(tmp_path / 'test.txt')
    data = np.random.rand(10, 10)
    meta = {}
    result = write_single_image(path, data, meta)
    assert result == [path]


def test_write_multiple(tmp_path):
    path = str(tmp_path / 'test_dir')
    data = [(np.random.rand(10, 10), {}, 'image')]
    result = write_multiple(path, data)
    assert result == [path]
