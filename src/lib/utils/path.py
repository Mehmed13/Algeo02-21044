from pathlib import Path
from typing import List
import os


def get_files(folder: str, image_cap: int):
    """Melakukan walk pada setiap subfolder dan mencari semua daftar gambar dengan ekstension jpg, png, atau jpeg

    Mengembalikan absolute path file
    """
    result: List[str] = []
    test_path = Path(os.getcwd()).joinpath(folder).resolve()

    counter = 0

    for root, _, files in os.walk(test_path, topdown=False):
        cap_reached = False
        for name in files:
            if name.endswith('.jpg') or name.endswith('.png') or name.endswith('.jpeg'):
                counter += 1
                if counter > image_cap:
                    cap_reached = True
                    break
                result.append(os.path.join(root, name))

        if cap_reached:
            break

    return result
