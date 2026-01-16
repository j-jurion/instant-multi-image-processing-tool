import pytest
from example import analyse_image
from imip import imip

NORMAL_CASE_IMAGE_PATH = "./images/normal"
EDGE_CASE_IMAGE_PATH = "./images/edge"

@pytest.mark.parametrize(
    "image_path, min_contours, max_contours, minimal_pass_rate",
    [
        (NORMAL_CASE_IMAGE_PATH, 0, 300, 0.9),
        (EDGE_CASE_IMAGE_PATH, 0, 300, 0.7),
    ],
)
def test_analyse_image(
    image_path: str,
    min_contours: int,
    max_contours: int,
    minimal_pass_rate: float,
) -> None:
    imip.load(image_path)
    results = imip.test_fn(analyse_image)
    print("Results:", results)
    assert all(result is not None for result in results)
    assert len([result for result in results if result is not None and min_contours <= result <= max_contours]) > minimal_pass_rate * len(results)
