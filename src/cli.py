import numpy as np
import matplotlib.pyplot as plt

from lib.utils import batch_load, load_image
from lib import normalize_image, mean_image, calculate_covariance, qr_algorithm, sort_image_by_eigenvalue, build_eigenfaces, match, calculate_weight

MAX_LOADED_IMAGE = 500

images, path = batch_load("../test/Dataset", MAX_LOADED_IMAGE)
image_count = len(images)

mean = mean_image(images)
normalized_images = normalize_image(images)

covariance = calculate_covariance(normalized_images)

eigenvalue, eigenvector = qr_algorithm(covariance)

eigenvalue_sorted, eigenvector_sorted, normalized_images_sorted, path_sorted = sort_image_by_eigenvalue(
    eigenvalue, eigenvector, normalized_images, path)

eigenfaces = build_eigenfaces(
    eigenvalue_sorted, eigenvector_sorted, normalized_images_sorted)

print(f"Eigenfaces selected {eigenfaces.__len__()}")

processed_image = calculate_weight(
    eigenfaces, normalized_images_sorted, path_sorted)

# bagian test
path, res = match("../test/Test/download (1).jpg", eigenfaces, processed_image, mean)
if(path==None):
    print(res)
    print("No match found")
else:
    print("Found at path "+ path+ " with " + str(round(res*100,2)) +"% matched")
# test_image = load_image("../test/Test/download (1).jpg")

# normalized_test = test_image - mean

# weight = []

# for eigenface in eigenfaces:
#     combination = eigenface.T @ normalized_test
#     weight.append(combination)

# result = []

# for image in processed_image:
#     distance = np.sqrt(np.power(np.array(image["weight"] - weight), 2).sum())
#     result.append(distance)

# minweight = min(result)

# idx = result.index(minweight)

# matched_image = processed_image[idx]

# print(f"Got path: {matched_image['path']}")