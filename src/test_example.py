import re
from example import analyse_image
from imip import imip

NORMAL_CASE_IMAGE_PATH = "./images/normal"
EDGE_CASE_IMAGE_PATH = "./images/edge"


def test_analyse_image():
    imip.load(NORMAL_CASE_IMAGE_PATH)
    imip.test_fn(analyse_image)
    assert all(result is not None for result in imip.results())
    assert len([result for result in imip.results() if result is not None and result > 60]) > 0.9 * len(imip.results())

