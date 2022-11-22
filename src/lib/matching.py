import numpy as np
import math
from typing import List
from .processing import Image
from .utils import load_image


def match(file_path, eigenfaces, processed_image: List[Image], mean):
    test_image = load_image(file_path)
    normalized_test = test_image - mean
    weight = []

    for eigenface in eigenfaces:
        combination = eigenface.T @ normalized_test
        weight.append(combination)

    # divider = np.array([np.sqrt(np.power(img['weight'], 2). sum()) for img in processed_image]).mean()

    result = []

    divider = np.sqrt(np.power(weight, 2).sum())

    for image in processed_image:
        distance = np.sqrt(
            np.power(np.array((image["weight"] - weight)/divider), 2).sum())
        # divider =

        result.append(1/(1+distance))
        # result.append(1-distance/divider)

    # maxweight = max(result)
    # divisor = math.sqrt(np.power(np.array(normalized_test), 2).sum())
    # # normalize result
    # result = [1-(a/divisor) for a in result]
    ret = max(result)

    if True:
        idx = result.index(ret)
        matched_image = processed_image[idx]
        closest_image_path = matched_image['path']
        return closest_image_path, ret
    return None, ret
