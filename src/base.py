from dataclasses import dataclass
import numpy as np

@dataclass
class ImageData:
    id: int
    image: np.ndarray

@dataclass
class ProcessedImageData(ImageData):
    original_image_id: int
    algorithm: str

@dataclass
class DataResult:
    data: np.ndarray
    description: str
