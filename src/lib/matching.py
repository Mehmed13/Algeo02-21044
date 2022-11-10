import numpy as np
from .utils import load_image


def macth(file_path, eigenfaces, eigenfaces_used, image_count, mean):
    test_image = load_image("../test/Alex Lawther102_3.jpg")
    normalized_test = test_image - mean

    weight = []
    for i in range(eigenfaces_used):
        combination = eigenfaces["image"][i].T @ normalized_test
        weight.append(combination)

    result = []

    for i in range(image_count):
        result.append(
            np.sqrt(np.power(np.array(eigenfaces["weight"][i]) - np.array(weight), 2)).sum())
    minweight = min(result)
    idx = result.index(minweight)
    return idx
