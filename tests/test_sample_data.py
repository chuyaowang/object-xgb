from object_xgb._sample_data import make_sample_data


def test_make_sample_data():
    data = make_sample_data()
    assert isinstance(data, list)
    assert len(data) > 0
    assert isinstance(data[0], tuple)
    assert data[0][0].shape == (512, 512)
