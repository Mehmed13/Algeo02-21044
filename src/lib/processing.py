import numpy as np
from typing import TypedDict, List
from .matrix import frobenious_form


class Image(TypedDict):
    normalized_image: np.ndarray
    path: np.ndarray
    weight: np.ndarray


def mean_image(array: np.ndarray):
    """Mendapatkan mean face gambar
    """

    return array.mean(dtype=np.float32, axis=0)


def normalize_image(array: np.ndarray):
    """Lakukan normalisasi gambar dengan langkah:

    Hitung matriks rata-rata gambar
    Kurangkan matriks rata-rata ke semua matriks gambar
    """
    return array - mean_image(array)


def calculate_covariance(images: np.ndarray):
    """Hitung kovarian matriks dengan mengalikan array matriks dengan transposenya
    """
    return images @ images.T


def sort_image_by_eigenvalue(eigenvalue: np.ndarray, eigenvector: np.ndarray, normalized_images: np.ndarray, path: np.ndarray):
    """Sort eigenvalue berdasarkan terbesar terlebih dahulu,
    lalu sort eigenvector, normalized_images, dan path berdasarkan
    hasil sort tadi
    """

    eigenpair = [(eigenvalue[i], eigenvector[:, i])
                 for i in range(normalized_images.__len__())]

    eigenpair.sort(reverse=True)

    sorted_index = sorted(
        range(eigenvalue.shape[0]), key=lambda k: eigenvalue[k], reverse=True)

    eigenvector_list = np.array([pair[1] for pair in eigenpair])

    eigenvalue_sorted = eigenvalue[sorted_index]
    eigenvector_sorted = eigenvector_list[sorted_index]
    normalized_images_sorted = normalized_images[sorted_index]
    path_sorted = path[sorted_index]

    return eigenvalue_sorted, eigenvector_sorted, normalized_images_sorted, path_sorted


def get_k_value(eigenvalue: np.ndarray):
    """Mengembalikan nilai k, dengan k adalah
    jumlah k eigenvalue pertama yang merepresentasikan K_TRESHOLD persen dari
    total eigenvalue
    """
    K_TRESHOLD = 0.95

    eigensum = eigenvalue.sum()
    current_sum = 0
    i = 0

    while current_sum/eigensum < K_TRESHOLD and i < eigenvalue.__len__():
        current_sum += eigenvalue[i]
        i += 1

    return i


def build_eigenfaces(eigenvalues: np.ndarray, eigenvector: np.ndarray, normalized_images: np.ndarray):
    """Membangun eigenfaces dari eigenvector, normalized_images, dst

    eigenvalue, vector, image diasumsikan sudah terurut secara menurun
    """

    k_value = get_k_value(eigenvalues)

    eigenfaces = []

    for i in range(k_value):
        vector = eigenvector[i]
        eigenface = vector @ normalized_images
        normal = frobenious_form(eigenface)
        eigenfaces.append(eigenface/normal)

    return eigenfaces


def calculate_weight(eigenfaces: np.ndarray, normalized_images: np.ndarray, path: np.ndarray):
    """Hitung kombindasi weight normalized image terhadap eigenfaces
    """

    result: List[Image] = []
    image_count = normalized_images.__len__()

    for i in range(image_count):
        weight = []

        for eigenface in eigenfaces:
            weight.append(eigenface.T @ normalized_images[i])

        result.append(Image(
            normalized_image=normalized_images[i],
            path=path[i],
            weight=np.array(weight)
        ))

    return result
