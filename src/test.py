import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import tkinter as tk
from lib.utils import batch_load, load_image
from lib import normalize_image, mean_image, calculate_covariance, qr_algorithm, frobenious_form


images, path = batch_load("../test/hgfhg")
image_count = len(images)

mean = mean_image(images)
normalized_images = normalize_image(images)
# mean = mean_image(images)
# # print(mean.shape)

# print(normalized_images[0])
# matplotlib.use('TkAgg')

# img = normalized_images[0]
# img = img - img.min()

# imgplot = plt.imshow(img.reshape((256, 256)), cmap='gray')
# imgplot = plt.imshow(mean.reshape((256, 256)), cmap='gray')
# plt.show()

covariance = calculate_covariance(normalized_images)
print(covariance.shape)


# time1 = time.time()
eigenval, eigenvector = qr_algorithm(covariance)

eigenpair = [(eigenval[i], eigenvector[:, i]) for i in range(image_count)]

eigenpair.sort(reverse=True)
# eigenval, eigenvector = np.linalg.eig(covariance)
# time2 = time.time()

# print(eigenval.shape)
# print(eigenvector.shape)
# print(f"Time taken {time2-time1}")
# print(eigenval)
# print(eigenvector)
# print(eigenval[:20].round(3))

# mx1 256^2x1
eigenfaces = {"image": [], "weight": []}

for i in range(image_count):
    efec = eigenpair[i][1]
    eigenface = efec@normalized_images
    # eigenface = eigenvector[:, i].T@normalized_images
    normal = frobenious_form(eigenface)
    eigenfaces["image"].append(eigenface/normal)


# print(eigenfaces["image"][0].shape)


eigenfaces_used = int(image_count/10) if image_count >= 100 else 5

for i in range(image_count):
    weight = []
    for j in range(eigenfaces_used):
        combination = eigenfaces["image"][j].T @ normalized_images[i]
        weight.append(combination)

    eigenfaces["weight"].append(weight)


for i in range(25):
    plt.subplot(5, 5, 1+i)
    plt.imshow(eigenfaces["image"][i].reshape((256, 256)), cmap='gray')

plt.show()


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
plt.subplot(5, 5, 1)
plt.imshow(normalized_test.reshape((256, 256)), cmap="gray")
# print(result)
minwieght = min(result)
idx = result.index(minwieght)
print(path[idx])
# print(idx)
# print(result)

plt.subplot(5, 5, 2)
plt.imshow(normalized_images[idx].reshape((256, 256)), cmap="gray")

plt.show()
