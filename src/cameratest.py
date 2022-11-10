import cv2

from lib import capture_image_from_camera, process_captured_image

try:
    result = capture_image_from_camera()
    result = process_captured_image(result)
    if result is not None:
        cv2.imwrite('crop.jpg', result)
except:
    pass
