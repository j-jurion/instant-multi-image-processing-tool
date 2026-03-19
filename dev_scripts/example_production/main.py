import cv2

from dev_scripts.example_production.product_analyzer import ProductAnalyzer


class ImageSaver:
    def debug(self, image, description=""):
        # Save image ...
        pass


global imip
imip = ImageSaver()


image = cv2.imread(
    "C:\\Users\\joeri\\Documents\\IMA\\images\\2026-03-06\\breed\\crops\\kleine_patch\\01_13.bmp"
)

if image is None:
    raise ValueError("Failed to load image. Please check the file path.")

product_analyzer = ProductAnalyzer()
product_analyzer.apply_shadow_detection_algorithm(
    image,
    red_multiplier=2,
    blue_multiplier=3,
)
