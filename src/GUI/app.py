import tkinter as tk
from tkinter import ttk  # new widgets and feature, themed widgets
from tkinter import filedialog as fd
# from ctypes import windl

# windl.shcore.SetProcessDpiAwareness(1)


def update_dataset(folder_choosen, dir_path):
    if (not (folder_choosen) or dir_path == ""):
        dataset_keterangan["text"] = "No Folder Choosen"
    else:
        dataset_keterangan["text"] = dir_path
    
    dataset_keterangan.pack()


def select_directory():
    dir_path = fd.askdirectory()
    folder_choosen = True
    update_dataset(folder_choosen, dir_path)


def update_file(file_choosen, file_path):
    if (not (file_choosen) or file_path == ""):
        file_keterangan["text"] = "No File Choosen"
    else:
        file_keterangan["text"] = file_path
    file_keterangan.pack()


def select_file():
    filetypes = (
        ('image files', '*.jpg, *.jpeg, *.png'),
        ('All files', '*.*')
    )

    file_path = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    file_choosen = True
    update_file(file_choosen, file_path)


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
app.iconbitmap(r'src\GUI\assets\keos.ico')  # set icon

# Header
head = tk.Text(app, height=1)
head.insert("1.0", "Face Recognition")
head.tag_add("header", "1.0", "2.0")
head.tag_configure("header", justify="center")
head.configure(font=("Helvetica", 30, "bold"),
               fg="#4649FF", state="disabled")
head.pack()

# Separator
separator = ttk.Separator(app, orient='horizontal')
separator.pack(fill='both')

# Data Section
# kontainer
data = tk.Frame(app)
data.pack()
# penanda belum ada folder yang dipilih
folder_choosen = False
dir_path = ""
# penanda belum ada file yang dipilih
file_choosen = False
file_path = ""

# dataset
dataset = tk.Frame(data)
dataset.pack(anchor=tk.NW)
dataset_text = tk.Label(
    dataset, text="Insert Your Dataset", font=("Helvetica", 15))
dataset_text.pack(anchor=tk.NW, ipadx=70, ipady=50)

dataset_button = ttk.Button(
    dataset, text="Choose Folder", command=select_directory)
dataset_button.pack(anchor=tk.W, ipadx=70, ipady=10)

dataset_keterangan = tk.Label(dataset, font=("Helvetica", 10))
update_dataset(folder_choosen, dir_path)

# file
file = tk.Frame(data)
file.pack(anchor=tk.W)
file_text = tk.Label(file, text="Insert Your file", font=("Helvetica", 15))
file_text.pack(anchor=tk.NW, ipadx=70, ipady=50)

file_button = ttk.Button(
    file, text="Choose File", command=select_file)
file_button.pack(anchor=tk.W, ipadx=70, ipady=10)

file_keterangan = tk.Label(file, font=("Helvetica", 10))
update_file(file_choosen, file_path)

# Header
if __name__ == "__main__":

    app.mainloop()
