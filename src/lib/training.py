import cv2 as cv2
import os
from .utils import image_loader
import numpy as np
from pathlib import Path
from typing import List


def normalize_array(array: np.ndarray):
    # mean = array.mean(dtype=np.int8,axis=0)
    mean = array.mean(dtype=np.float32, axis=0)
    print(f"mean shape {mean.shape}")
    # print("MEAN")
    # print(mean)
    return (array - mean.astype(np.float16)), mean


def getcovariance(normalized):
    return (np.asarray(
        [np.matmul(arr.astype(np.float32), arr.transpose().astype(
            np.float32), dtype=np.float32) for arr in normalized],
    ).mean(axis=0, dtype=np.float32))


def training(folder):
    images, images_path = image_loader.batch_load(folder, absolute=True)
    image_count = len(images)
    normalized, mean = normalize_array(images)
    images = None
    covariance = getcovariance(normalized)
    return covariance, image_count, normalized, mean, images_path
