import numpy as np


def frobenious_form(A: np.ndarray):
    """Menghitung normal matriks

    Referensi:
    https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
    """
    return np.sqrt(np.power(A, 2).sum())


def qr_factorization_householder(A: np.ndarray):
    """Melakukan dekomposisi QR menggunakan householder

    Referensi:
    https://github.com/hbldh/b2ac/blob/master/b2ac/matrix/matrix_algorithms.py
    """
    A = np.array(A, np.float64)

    n, m = A.shape
    V = np.zeros_like(A, np.float64)

    for k in range(n):
        V[k:, k] = A[k:, k].copy()
        V[k, k] += np.sign(V[k, k]) * frobenious_form(V[k:, k])
        V[k:, k] /= frobenious_form(V[k:, k])
        A[k:, k:] -= 2 * np.outer(V[k:, k], np.dot(V[k:, k], A[k:, k:]))

    R = np.triu(A[:n, :n])

    Q = np.eye(m, n)

    for k in range((n - 1), -1, -1):
        Q[k:, k:] -= np.dot((2 * (np.outer(V[k:, k], V[k:, k]))), Q[k:, k:])

    return Q, R


def householder_vectorized(a):
    """Menghitung vector householder

    Referensi:
    https://stackoverflow.com/questions/53489237/how-can-you-implement-householder-based-qr-decomposition-in-python
    """
    v = a / (a[0] + np.copysign(frobenious_form(a), a[0]))
    v[0] = 1
    tau = 2 / (v.T @ v)

    return v, tau


def hessenberg_reduction(A: np.ndarray):
    """Menghitung matriks hessenberg. Melakukan reduksi matriks A menjadi matriks Hessenberg

    Referensi:
    https://www.math.usm.edu/lambers/mat610/class0329.pdf
    """
    m = A.shape[0]

    U = np.eye(m)
    H = A.copy().astype(np.float64)

    for j in range(m-2):
        v, c = householder_vectorized(H[j+1:, j, np.newaxis])
        H[j+1:, j:] = H[j+1:, j:] - c * v @ v.transpose() @ H[j+1:, j:]
        H[0:, j+1:] = H[0:, j+1:] - H[0:, j+1:] @ ((c*v) @ v.transpose())
        U[0:, j+1:] = U[0:, j+1:] - U[0:, j+1:] @ ((c*v) @ v.transpose())

    return H, U


def qr_algorithm(A: np.ndarray):
    """Melakukan QR Algorithm dengan QR decomposition secara eksplisit

    Matriks A dikonversi terlebih dahulu diubah ke matirks Hessenberg

    Referensi:
    https://towardsdatascience.com/eigenvalues-and-eigenvectors-89483fb56d56
    https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf
    """
    H, _ = hessenberg_reduction(A)  # reduksi ke matriks hessenberg
    # QR factorization dengan householder
    Q, _ = qr_factorization_householder(H)

    E = Q.T @ A @ Q
    U = Q

    result = np.diag(E)

    i = 0

    # todo: buat metrik untuk berhenti ketika perubahan setiap siklusnya sudah cukup kecil
    while i < 50:
        Q, _ = qr_factorization_householder(E)
        E = Q.T @ E @ Q
        U = U @ Q
        i += 1
        result = np.diag(E)

    return result, U
