import numpy as np
import matplotlib.pyplot as plt

from lib.utils import batch_load, load_image
from lib import normalize_image, mean_image, calculate_covariance, qr_algorithm, sort_image_by_eigenvalue, build_eigenfaces, calculate_weight

MAX_LOADED_IMAGE = 500

images, path = batch_load("../test/hgfhg", MAX_LOADED_IMAGE)
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
test_image = load_image("../test/elon musk157_1557.jpg")

normalized_test = test_image - mean

weight = []

for eigenface in eigenfaces:
    combination = eigenface.T @ normalized_test
    weight.append(combination)

result = []

for image in processed_image:
    distance = np.sqrt(np.power(np.array(image["weight"] - weight), 2).sum())
    result.append(distance)

minweight = min(result)

idx = result.index(minweight)

matched_image = processed_image[idx]

print(f"Got path: {matched_image['path']}")

plt.subplot(2, 2, 1)
plt.imshow(normalized_test.reshape((256, 256)), cmap='gray')
plt.subplot(2, 2, 2)
plt.imshow(matched_image["normalized_image"].reshape((256, 256)), cmap='gray')
plt.show()
