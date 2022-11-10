import cv2  # type: ignore
import numpy as np

from .path import get_files


def load_image(path: str) -> np.ndarray:
    """Load gambar dengan menggunakan opencv
    Konversi gambar menjadi grayscale dan direduksi ke resolusi 256x256 pixel
    """
    resized = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (256, 256))

    return np.asarray(resized, np.uint8).flatten()


def batch_load(folder: str, image_cap=2000, absolute=False):
    """Load banyak gambar sekaligus dari folder dataset
    """
    image_path = get_files(folder, image_cap, absolute)

    return np.asarray([load_image(path) for path in image_path], np.uint8), image_path
