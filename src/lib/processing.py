import numpy as np


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
