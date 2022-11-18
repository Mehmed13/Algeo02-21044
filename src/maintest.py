import tkinter as tk
from tkinter import ttk  # new widgets and feature, themed widgets
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import time
from lib.utils import batch_load, load_image
from lib import normalize_image, mean_image, calculate_covariance, qr_algorithm, sort_image_by_eigenvalue, build_eigenfaces, calculate_weight, matching, capture_image_from_camera, process_captured_image
import cv2

app = tk.Tk()
window_height = 720
window_width = 1080

screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
app.title('Keos Recognition')

center_x = int(screen_width/2 - window_width/2)
center_y = int(screen_height/2 - window_height/2)

# set the window size
app.geometry(f'{window_width}x{window_height}+{center_x}-{center_y}')
app.resizable(False, False)  # tidak bisa di resize
app.iconbitmap(r'GUI\assets\keos.ico')  # set icon
app.columnconfigure(1, weight=1)
app.columnconfigure(1,weight=3)
head = tk.Label(app, text="Face Recognition", justify="center",
                font=("Helvetica", 30, "bold"))
head.grid(row=0, column=0, columnspan=3, pady=30)
data = tk.Frame(app, bg='red')


data.grid(row=1, column=0,ipady=10)
dataset_text = tk.Label(
    data, text="Insert Your Dataset", font=("Helvetica", 15))
dataset_text.grid(row=0,column=0)


data2 = tk.Frame(app, bg='green')


data2.grid(row=1, column=1,ipady=10, sticky=tk.EW)
data2.columnconfigure(0, weight=2)
dataset2_text = tk.Label(
    data2, text="Insert Your Dataset", font=("Helvetica", 15))
# data2.grid_propagate(False)
dataset2_text.grid(row=0,column=0)
# penanda belum ada folder yang dipilih
folder_choosen = False
dir_path = ""
# penanda belum ada file yang dipilih
file_choosen = False
file_path = ""
# dataset
# dataset = tk.Frame(data)
# dataset.pack(ipady=10)
# dataset_text = tk.Label(
#     dataset, text="Insert Your Dataset", font=("Helvetica", 15))
# dataset_text.pack(ipady=20)

# dataset_button = ttk.Button(
#     dataset, text="Choose Folder")
# dataset_button.pack(ipadx=70, ipady=10)

# dataset_keterangan = tk.Label(dataset, font=("Helvetica", 10))


# # file
# file = tk.Frame(data)
# file.pack(ipady=10)
# file_text = tk.Label(file, text="Insert Your File", font=("Helvetica", 15))
# file_text.pack(ipadx=70, ipady=20)

# file_button = ttk.Button(
#     file, text="Choose File")
# file_button.pack(ipadx=70, ipady=10)

# file_button = ttk.Button(
#     file, text="Use Camera")
# file_button.pack(ipadx=70, ipady=10)

# file_keterangan = tk.Label(file, font=("Helvetica", 10))


# # result
# result = tk.Frame(data)
# result.pack(ipady=10)

# result_text = tk.Label(result, text="Result", font=("Helvetica", 18))
# result_text.pack(ipadx=70, ipady=30)

# result_keterangan = tk.Label(result, font=("Helvetica", 10))
# result_path = tk.Label(result, font=("Helvetica", 10), wraplength=300)
# res = 0
# match = False  # inisialisasi
# execute = False
# prev_dir = ""

# frame2 = tk.Frame(app, bg='green')
# frame2.grid(column=1,row=1)
# frame2_text = tk.Label(
#     frame2, text="Insert Your Dataset", font=("Helvetica", 15))
# frame2_text.grid(row=0,column=0)

# frame1 = tk.LabelFrame(app, text="Fruit", bg="green",
#                     fg="white", padx=15, pady=15)
  
# # Displaying the frame1 in row 0 and column 0
# frame1.grid(row=0, column=0)
# head2 = tk.Label(app, text="Face Recognition", justify="center",
#                 font=("Helvetica", 30, "bold"))
# head2.grid(row=1, column=1, pady=30)
# head2 = tk.Label(app, text="Face Recognition", justify="center",
#                 font=("Helvetica", 30, "bold"))
# head2.grid(row=1, column=2, pady=30)

if __name__ == "__main__":

    app.mainloop()
