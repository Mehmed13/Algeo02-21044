import cv2 as cv2
import os
import numpy as np
from pathlib import Path
from typing import List


def get_files(folder: str):
    result: List[str] = []
    test_path = Path(os.getcwd()).joinpath(folder)

    counter = 0
    image_cap = 2000

    for root, _, files in os.walk(test_path, topdown=False):
        for name in files:
            if name.endswith('.jpg') or name.endswith('.png') or name.endswith('.jpeg'):
                counter+=1
                if counter > image_cap:
                    break
                result.append(os.path.join(root, name))

    return result

# load image 

def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(image, (256,256))
    # print(f"Loading image with path {path}")
    return np.asarray(resized, np.uint8)


def batch_load(folder: str):
    image_paths = get_files(folder)
    return np.asarray([load_image(path) for path in image_paths], np.uint8)


def normalize_array(array: np.ndarray):
    # mean = array.mean(dtype=cp.int8,axis=0)
    mean = array.mean(dtype=np.float32, axis=0)
    print(f"mean shape {mean.shape}")
    # print("MEAN")
    # print(mean)
    return array - mean.astype(np.float16)

def load_data(training):
    folder = input("Masukkan nama folder: ")
    