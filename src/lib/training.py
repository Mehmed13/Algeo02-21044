import cv2 as cv2
import os
import utils.image_loader as loader
import numpy as np
from pathlib import Path
from typing import List


def normalize_array(array: np.ndarray):
    # mean = array.mean(dtype=np.int8,axis=0)
    mean = array.mean(dtype=np.float32, axis=0)
    print(f"mean shape {mean.shape}")
    # print("MEAN")
    # print(mean)
    return array - mean.astype(np.float16)

def getcovariance(normalized):
    return (np.asarray(
    [np.matmul(arr.astype(np.float32), arr.transpose().astype(np.float32), dtype=np.float32) for arr in normalized],
).mean(axis=0, dtype=np.float32))
    
def training():
    folder = input("Input Training Images Folder Name: ")
    images = loader.batch_load(folder)
    normalized: np.ndarray = normalize_array(images)
    images = None
    covariance = getcovariance(normalized)
    