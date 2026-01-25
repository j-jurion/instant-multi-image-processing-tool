import asyncio
import os
from pathlib import Path
from typing import Any, Callable

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from heliovision.streams.stream import Observable, Stream
from loguru import logger
from viewer import Viewer
from base import ImageBundle



class IMIP:
    def __init__(self):
        self.debug_images: list[ImageBundle] = []
        self.current_image_index: int | None = None
        self.output_directory = None
        self.save_debug_images = False
        self.imip_reloader: IMIPReloader | None = None
        self.viewer = Viewer(self.debug_images)

    def set_debug_save_dir(self, output_directory: Path | None) -> None:
        if output_directory is None:
            self.save_debug_images = False
        else:
            self.output_directory = output_directory
            self.save_debug_images = True

    def load_images(self, path: Path) -> None:
        if path.is_file():
            self._load_image(path)
        elif path.is_dir():
            for img_file in path.glob("*.*"):
                if img_file.is_file():
                    self._load_image(img_file)
        else:
            raise ValueError(f"Path {path} is neither a file nor a directory.")

    def _load_image(self, path: Path) -> None:
        try:
            image = cv.imread(str(path))
            if image is None:
                logger.warning(f"Image at {path} could not be loaded. Image is None.")
            else:
                self.debug_images.append(
                    ImageBundle(source_image=image, processed_images={})
                )
                self.viewer.update(self.debug_images)
        except Exception as e:
            logger.warning(f"Could not load image {path}: {e}")

    async def debug_fn(self, function: Callable[[Any], Any], watch_file: Path) -> None:
        self.imip_reloader = IMIPReloader(watch_file)

        def run_function(reloaded: bool):
            if reloaded:
                logger.debug("File reloaded")
                for i, image_bundle in enumerate(self.debug_images):
                    logger.debug(f"Processing image {i + 1}/{len(self.debug_images)}")
                    self.current_image_index = i
                    logger.debug(image_bundle)
                    function(image_bundle.source_image)
                self.viewer.update(self.debug_images)

        self.imip_reloader.file_reloaded_stream.subscribe(
            on_next=run_function,
            on_error=lambda e: logger.error(e),
        )

        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.imip_reloader.run())

    def debugger(self, image: np.ndarray, description: str = ""):
        assert self.current_image_index is not None, "No current image data set."
        self.debug_images[self.current_image_index].processed_images[description] = (
            image
        )
        self.viewer.update(self.debug_images)

    def file_is_reloaded(self, filepath: Path) -> bool:
        current_mtime = os.path.getmtime(filepath)
        if not hasattr(self, "_file_mtimes"):
            self._file_mtimes = {}
        last_mtime = self._file_mtimes.get(filepath, None)
        self._file_mtimes[filepath] = current_mtime
        return last_mtime is None or current_mtime != last_mtime



class IMIPReloader:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._file_reloaded_stream: Stream[bool] = Stream()

    @property
    def file_reloaded_stream(self) -> Observable[bool]:
        return self._file_reloaded_stream

    async def run(self):
        while True:
            # Process matplotlib GUI events
            plt.pause(0.001)
            
            current_mtime = os.path.getmtime(self.filepath)
            if not hasattr(self, "_file_mtimes"):
                self._file_mtimes = {}
            last_mtime = self._file_mtimes.get(self.filepath, None)
            self._file_mtimes[self.filepath] = current_mtime
            if last_mtime is None or current_mtime != last_mtime:
                self._file_reloaded_stream.next(True)
            await asyncio.sleep(0.01)
