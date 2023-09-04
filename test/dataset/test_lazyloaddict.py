import pytest

from supervision.dataset.utils import LazyLoadDict

from .test_core import TEST_IMG_PATH


def test_lazy_load_dict_get_item(tmp_path):
    ld = LazyLoadDict({"test_img": TEST_IMG_PATH})
    img = ld["test_img"]
    assert img.shape == (100, 100, 3)


def test_lazy_load_dict_set_item():
    ld = LazyLoadDict()
    ld["new_img"] = TEST_IMG_PATH
    assert ld["new_img"].shape == (100, 100, 3)


def test_lazy_load_dict_del_item():
    ld = LazyLoadDict({"img1": TEST_IMG_PATH})
    del ld["img1"]
    with pytest.raises(KeyError):
        ld["img1"]


def test_lazy_load_dict_len():
    ld = LazyLoadDict({"img1": TEST_IMG_PATH, "img2": TEST_IMG_PATH})
    assert len(ld) == 2
