from dataclasses import dataclass, field
import os
from turtle import update 
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class DebugImage:
    image: np.ndarray
    description: str = ""


@dataclass
class ImageData:
    index: int
    image: np.ndarray
    debug_images: list[DebugImage] = field(default_factory=list)
    result_image: np.ndarray | None = None
    result_data: dict | None = field(default_factory=dict)


class IMIP:
    def __init__(self):
        self.data: list[ImageData] = []
        self.current_data_index: int | None = None
        self.output_directory: str = ""
        self.debug_images_saved: bool = False
        
    def load(self, directory: str):
        for file in os.listdir(directory):
            if file.endswith('.bmp') or file.endswith('.png') or file.endswith('.jpg'):
                filepath = os.path.join(directory, file)
                image = cv.imread(filepath)
                if image is not None:
                    index = len(self.data)
                    self.data.append(ImageData(index=index, image=image))

    def loaded_images(self) -> list[ImageData]:
        return self.data
    
    def register_result(self, index: int, result: np.ndarray, data: dict):
        for img_data in self.data:
            if img_data.index == index:
                img_data.result_image = result
                img_data.result_data = data
                break
    
    #
    # Debugger functions -----------------------------------------------------------
    #

    def debugger(self, image: np.ndarray, description: str = ""):
        assert self.current_data_index is not None, "No current image data set."
        self.data[self.current_data_index].debug_images.append(DebugImage(image=image, description=description))
        if self.debug_images_saved:
            self.save_debug_images(self.output_directory)

    def show_debug_images(self):
        assert self.current_data_index is not None, "No current image data set."
        for img_data in self.data:
            for i, debug_image in enumerate(img_data.debug_images):
                cv.imshow(f"Image {img_data.index} - {i}: {debug_image.description}", debug_image.image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def set_debugger_options(self, output_directory: str, debug_images_saved: bool):
        self.output_directory = output_directory
        self.debug_images_saved = debug_images_saved

    def save_debug_images(self, output_directory: str):
        assert self.current_data_index is not None, "No current image data set."
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        for img_data in self.data:
            for i, debug_image in enumerate(img_data.debug_images):
                output_path = os.path.join(output_directory, f"Image_{img_data.index}_Debug_{i}.png")
                cv.imwrite(output_path, debug_image.image)

    #
    # Instant processing functions -----------------------------------------------------------
    #

    def debug_fn(self, function):
        for i, image_data in enumerate(self.loaded_images()):
            self.current_data_index = i
            function(image_data.image)
        self.show_debug_images()
    
imip = IMIP()

