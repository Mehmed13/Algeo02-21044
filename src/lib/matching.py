import numpy as np
import math
from .utils import load_image


def match(file_path, eigenfaces, processed_image, mean):
    test_image = load_image(file_path)
    normalized_test = test_image - mean
    weight = []

    for eigenface in eigenfaces:
        combination = eigenface.T @ normalized_test
        weight.append(combination)

    result = []

    for image in processed_image:
        distance = np.sqrt(
            np.power(np.array(image["weight"] - weight), 2).sum())
        result.append(distance)

    maxweight = max(result)
    divisor = math.sqrt(np.power(np.array(result), 2).sum())
    # normalize result
    result = [1-(a/divisor) for a in result]
    ret = max(result)

    if ret>=0.5:
        idx = result.index(ret)
        matched_image = processed_image[idx]
        closest_image_path = matched_image['path']
        return closest_image_path, ret
    return None, ret