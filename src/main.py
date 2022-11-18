import tkinter as tk
from tkinter import ttk  # new widgets and feature, themed widgets
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import time
from lib.utils import batch_load, load_image
from lib import normalize_image, mean_image, calculate_covariance, qr_algorithm, sort_image_by_eigenvalue, build_eigenfaces, calculate_weight, matching, capture_image_from_camera, process_captured_image
import cv2
# from ctypes import windl

# windl.shcore.SetProcessDpiAwareness(1)

global match
global res

def select_directory():
    global dir_path
    dir_path = fd.askdirectory()
    folder_choosen = True
    update_dataset(folder_choosen)


def update_dataset(folder_choosen):
    if (not (folder_choosen) or dir_path == ""):
        dataset_keterangan["text"] = "No Folder Choosen"
    else:
        dataset_keterangan["text"] = "Folder Choosen"

    dataset_keterangan.pack()


def update_file(file_choosen):
    if (not (file_choosen) or file_path == ""):
        if (file_path == ""):
            file_choosen = False
        file_keterangan["text"] = "No File Choosen"
    else:
        file_keterangan["text"] = "File Choosen"
    file_keterangan.pack(ipadx=70)

def use_camera():
    global file_path
    try:
        result = capture_image_from_camera()
        if result is not None:
            cv2.imwrite('../test/Test/camera.jpg', result)
        file_path = '../test/Test/camera.jpg'
        file_choosen = True
        update_file(file_choosen)
        update_test_image(file_choosen)
        update_closest_image(False)
        update_exec_time(False)
        update_result(False, 0)
    except:
        pass

def select_file():
    global file_path
    filetypes = (
        ('image files', '*.jpg'),
        ('image files', '*.jpeg'),
        ('image files', '*.png'),
        ('All files', '*.*')
    )
    prev_file_path = file_path
    file_path = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    file_choosen = True
    if (file_path == ""):
        file_path = prev_file_path
    update_file(file_choosen)
    update_test_image(file_choosen)
    update_closest_image(False)
    update_exec_time(False)
    update_result(False, 0)


def update_result(match, res):
    if (match):
        result_keterangan["text"] = str(round(res*100, 2)) +"% Match"
        result_path["text"] = closest_image_path
    else:
        result_keterangan["text"] = "None"
        result_path["text"] = ""
    result_keterangan.pack(anchor=tk.N)
    result_path.pack(anchor=tk.N)


def update_test_image(file_choosen):
    global test_img
    if (file_choosen and file_path != ""):
        image = Image.open(file_path)
    else:
        if (file_path == ""):
            file_choosen = False
        image = Image.open('GUI/assets/blank_image.png')
    img = image.resize((300, 300))
    test_img = ImageTk.PhotoImage(img)
    test_image["image"] = test_img
    test_image.pack()


def update_exec_time(execute):
    if (execute):
        exec_time["text"] = "Execution time:{execution_time:.2f} s".format(
            execution_time=execution_time)
    else:
        exec_time["text"] = "Execution time:00.00"
    exec_time.pack(ipadx=30, pady=20)


def recognize():
    global closest_image_path, eigenfaces, processed_image, mean, execution_time, prev_dir
    # Memastikan agar sudah terdapat test_image dan folder dataset
    if (file_path != "" and dir_path != ""):
        start_time = time.time()
        if (prev_dir != dir_path):  # lakukan training jika berganti dataset
            # Membentuk matriks gambar dan array of image_path
            images, images_path = batch_load(dir_path, absolute=True)
            image_count = len(images)  # Menghitung banyak gambar dataset
            mean = mean_image(images)  # Menghitung rata-rata matriks gambar
            normalized_images = normalize_image(images)  # normalisasi
            covariance = calculate_covariance(
                normalized_images)  # Menghitung matriks kovarian
            # Menghitung eigenvalue dan eigen vector dari matriks kovarian
            eigenvalue, eigenvector = qr_algorithm(covariance)
            eigenvalue_sorted, eigenvector_sorted, normalized_images_sorted, images_path_sorted = sort_image_by_eigenvalue(
                eigenvalue, eigenvector, normalized_images, images_path)
            eigenfaces = build_eigenfaces(
                eigenvalue_sorted, eigenvector_sorted, normalized_images_sorted)
            processed_image = calculate_weight(
                eigenfaces, normalized_images_sorted, images_path_sorted)
        closest_image_path, res = matching.match(
            file_path, eigenfaces, processed_image, mean)
        if(closest_image_path!=None):
            match = True
            update_closest_image(match)
            update_result(match, res)
        else:
            match = False
        execute = True
        execution_time = time.time() - start_time
        update_exec_time(execute)
        update_result(match, res)
    prev_dir = dir_path


def update_closest_image(match):
    global closest_img
    if (match):
        image = Image.open(closest_image_path)
    else:
        image = Image.open('GUI/assets/blank_image.png')
    img = image.resize((300, 300))
    closest_img = ImageTk.PhotoImage(img)
    closest_image["image"] = closest_img
    closest_image.pack()


app = tk.Tk()

# set dimension variable

window_height = 720
window_width = 1080

screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()

center_x = int(screen_width/2 - window_width/2)
center_y = int(screen_height/2 - window_height/2)

# keep the window displaying
app.title('Keos Recognition')

# set the window size
app.geometry(f'{window_width}x{window_height}+{center_x}-{center_y}')
app.resizable(False, False)  # tidak bisa di resize
app.iconbitmap(r'GUI\assets\keos.ico')  # set icon
app.columnconfigure(0, weight=1)
app.columnconfigure(1, weight=2)
app.columnconfigure(2,weight=2)

# Header
head = tk.Label(app, text="Face Recognition", justify="center",
                font=("Helvetica", 30, "bold"))
head.grid(row=0, column=0, columnspan=4, pady=30)
# Separator


# Data Section
# kontainer
data = tk.Frame(app)
data.grid(row=1, column=0,padx=(70,20), ipady=10)
# penanda belum ada folder yang dipilih
folder_choosen = False
dir_path = ""
# penanda belum ada file yang dipilih
file_choosen = False
file_path = ""
# dataset
dataset = tk.Frame(data)
dataset.pack(ipady=10)
dataset_text = tk.Label(
    dataset, text="Insert Your Dataset", font=("Helvetica", 15))
dataset_text.pack(ipady=20)

dataset_button = ttk.Button(
    dataset, text="Choose Folder", command=select_directory)
dataset_button.pack(ipadx=70, ipady=10)

dataset_keterangan = tk.Label(dataset, font=("Helvetica", 10))
update_dataset(folder_choosen)

# file
file = tk.Frame(data)
file.pack()
file_text = tk.Label(file, text="Insert Your File", font=("Helvetica", 15))
file_text.pack(ipady=20)

file_button = ttk.Button(
    file, text="Choose File", command=select_file)
file_button.pack(ipadx=70, ipady=10)

file_button = ttk.Button(
    file, text="Use Camera", command=use_camera)
file_button.pack(ipadx=70, ipady=10, pady=10)

file_keterangan = tk.Label(file, font=("Helvetica", 10))
update_file(file_choosen)

# result
result = tk.Frame(data)
result.pack()

result_text = tk.Label(result, text="Result", font=("Helvetica", 15))
result_text.pack(ipady=20)

result_keterangan = tk.Label(result, font=("Helvetica", 10), fg='green')
result_path = tk.Label(result, font=("Helvetica", 10), wraplength=200)
res = 0
match = False  # inisialisasi
execute = False
prev_dir = ""
update_result(match, res)


display_image = tk.Frame(app)
display_image.grid(row=1, column=1,padx=(20,70))
display_image.columnconfigure(0, weight=1)
display_image.columnconfigure(1, weight=1)
# Test Image
test = tk.Frame(display_image)
test.grid(row=1, column=0, padx=20)

test_text = tk.Label(test, text="Test Image", font=("Helvetica", 10))
test_text.pack(pady=30)

test_image = tk.Label(test)
update_test_image(file_choosen)

recognize_button = ttk.Button(test, text="Recognize Image", command=recognize)
recognize_button.pack(ipadx=70, ipady=10, pady=10)

# Closest Result
closest = tk.Frame(display_image)
closest.grid(column=1, row=1, padx=20)

closest_text = tk.Label(closest, text="Closest Result",
                        font=("Helvetica", 10))
closest_text.pack(pady=30)

closest_image = tk.Label(closest)
update_closest_image(match)
exec_time = tk.Label(closest, font=("Helvetica", 10))
update_exec_time(execute)


if __name__ == "__main__":

    app.mainloop()
