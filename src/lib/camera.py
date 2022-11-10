import cv2
import numpy as np


def process_captured_image(image: np.ndarray):
    """Ubah gambar menjadi grayscale, lalu resize menjadi ukuran 256,256
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.resize(image_gray, (256, 256))

    return image_gray


def capture_image_from_camera():
    """Panggil webcam, lalu deteksi wajah
    Masukkan input q untuk keluar dan mengambil frame terakhir dari wajah yang pertama terdeteksi
    Mengembalikan array gambar yang terdeteksi, jika ada

    Periksa return value fungsi ini apakah None
    Catch juga setiap expection yang keluar dari fungsi ini
    """
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 640)
    video_capture.set(4, 480)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    result = None

    while True:
        success, img = video_capture.read()

        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_classifier.detectMultiScale(image_gray, 1.3, 5)

        cropped_image = None

        if faces.__len__() != 0:
            (x, y, w, h) = faces[0]

            outer_offset = 5

            if w > h:
                offset = int((w-h)/2)
                cropped_image = img[y-offset-outer_offset:y+h+offset +
                                    outer_offset, x-outer_offset:x+w+outer_offset].copy()
            elif h > w:
                offset = int((h-w)/2)
                cropped_image = img[y-outer_offset:y+h+outer_offset,
                                    x-offset-outer_offset, x+w+offset+outer_offset].copy()
            else:
                cropped_image = img[y-outer_offset:y+h+outer_offset,
                                    x-outer_offset:x+w+outer_offset].copy()

            img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

        cv2.imshow('face_detector', img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            result = cropped_image
            break

    video_capture.release()
    cv2.destroyWindow('face_detector')

    return result
