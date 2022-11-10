import tkinter as tk
from tkinter import ttk  # new widgets and feature, themed widgets
from tkinter import filedialog as fd
from PIL import Image, ImageTk
from lib import training, matrix, matching
# from ctypes import windl

# windl.shcore.SetProcessDpiAwareness(1)


def select_directory():
    global dir_path
    dir_path = fd.askdirectory()
    folder_choosen = True
    update_dataset(folder_choosen)


def update_dataset(folder_choosen):
    if (not (folder_choosen) or dir_path == ""):
        if (dir_path == ""):
            folder_choosen = False
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
    file_keterangan.pack()


def select_file():
    global file_path
    filetypes = (
        ('image files', '*.jpg, *.jpeg, *.png'),
        ('All files', '*.*')
    )

    file_path = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    file_choosen = True
    update_file(file_choosen)
    update_test_image(file_choosen)


def update_result(match):
    if (match):
        result_keterangan["text"] = "Matched"
    else:
        result_keterangan["text"] = "None"
    result_keterangan.pack(anchor=tk.N)


def update_test_image(file_choosen):
    global test_img
    if (file_choosen and file_path != ""):
        image = Image.open(file_path)
    else:
        if (file_path == ""):
            file_choosen = False
        image = Image.open('GUI/assets/blank_image.png')
    img = image.resize((350, 350))
    test_img = ImageTk.PhotoImage(img)
    test_image["image"] = test_img
    test_image.pack()


def update_exec_time(execute):
    if (execute):
        exec_time["text"] = "Execution time:MM.DD"
    else:
        exec_time["text"] = "Execution time:00.00"
    exec_time.pack(ipady=15)


def recognize():
    global closest_image_path, eigenfaces, eigenfaces_used, mean, image_count
    if (file_path != ""):  # Memastikan agar sudah terdapat test_image
        if (prev_dir != dir_path):  # lakukan training jika berganti dataset
            covariance, image_count, normalized_images, mean, images_path = training.training(
                dir_path)  # Mencari matriks kovarian
            eigenval, eigenvector = matrix.qr_algorithm(covariance)
            eigenpair = [(eigenval[i], eigenvector[:, i])
                         for i in range(image_count)]
            eigenpair.sort(reverse=True)
            # mx1 256^2x1
            eigenfaces = {"image": [], "weight": []}

            for i in range(image_count):
                efec = eigenpair[i][1]
                eigenface = efec@normalized_images
                # eigenface = eigenvector[:, i].T@normalized_images
                normal = matrix.frobenious_form(eigenface)
                eigenfaces["image"].append(eigenface/normal)

            eigenfaces_used = int(image_count/10) if image_count >= 100 else 5

            for i in range(image_count):
                weight = []
                for j in range(eigenfaces_used):
                    combination = eigenfaces["image"][j].T @ normalized_images[i]
                    weight.append(combination)

                eigenfaces["weight"].append(weight)
        closest_image_idx = matching.macth(
            file_path, eigenfaces, eigenfaces_used, image_count, mean)
        closest_image_path = images_path[closest_image_idx]
        match = True
        update_closest_image(match)


def update_closest_image(match):
    global closest_img
    if (match):
        image = Image.open(closest_image_path)
    else:
        image = Image.open('GUI/assets/blank_image.png')
    img = image.resize((350, 350))
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


# Header
head = tk.Label(app, text="Face Recognition", justify="center",
                font=("Helvetica", 30, "bold"))
head.grid(row=0, column=0, columnspan=3, pady=30)
# Separator


# Data Section
# kontainer
data = tk.Frame(app)
data.grid(row=2, column=0, sticky=tk.SW, pady=10)
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
dataset_text.pack(ipadx=70, ipady=30)

dataset_button = ttk.Button(
    dataset, text="Choose Folder", command=select_directory)
dataset_button.pack(ipadx=70, ipady=10)

dataset_keterangan = tk.Label(dataset, font=("Helvetica", 10))
update_dataset(folder_choosen)

# file
file = tk.Frame(data)
file.pack(ipady=10)
file_text = tk.Label(file, text="Insert Your file", font=("Helvetica", 15))
file_text.pack(ipadx=70, ipady=30)

file_button = ttk.Button(
    file, text="Choose File", command=select_file)
file_button.pack(ipadx=70, ipady=10)

file_keterangan = tk.Label(file, font=("Helvetica", 10))
update_file(file_choosen)

# result
result = tk.Frame(data)
result.pack(ipady=10)

result_text = tk.Label(result, text="Result", font=("Helvetica", 18))
result_text.pack(anchor=tk.NW, ipadx=70, ipady=30)

result_keterangan = tk.Label(result, font=("Helvetica", 10))

match = False  # inisialisasi
execute = False
prev_dir = ""
update_result(match)

# Test Image
test = tk.Frame(app)
test.grid(row=2, column=1, sticky=tk.S, pady=10)

test_text = tk.Label(test, text="Test Image", font=("Helvetica", 10))
test_text.pack(ipadx=70, ipady=25)

test_image = tk.Label(test)
update_test_image(file_choosen)

recognize_button = ttk.Button(test, text="Recognize Image", command=recognize)
recognize_button.pack(ipadx=70, pady=10, ipady=5)

# Closest Result
closest = tk.Frame(app)
closest.grid(column=2, row=2, sticky=tk.SE, pady=10)

closest_text = tk.Label(closest, text="Closest Result",
                        font=("Helvetica", 10))
closest_text.pack(ipadx=70, ipady=30)

closest_image = tk.Label(closest)
update_closest_image(match)
exec_time = tk.Label(closest, font=("Helvetica", 7))
update_exec_time(execute)


if __name__ == "__main__":

    app.mainloop()
