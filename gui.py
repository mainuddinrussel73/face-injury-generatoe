import tkinter as tk
from tkinter import filedialog
from main import apply_injury_multiple  # Import the function from the main script

def open_image():
    global img_path
    img_path = filedialog.askopenfilename(title="Select Face Image", filetypes=[("Image Files", "*.jpg *.png")])
    if img_path:
        label.config(text="Selected: " + img_path.split('/')[-1])

def apply_injury_gui():
    if img_path:
        injuries = injury_listbox.curselection()  # Get selected injuries
        selected_injuries = [injury_var.get(i) for i in injuries]
        apply_injury_multiple(img_path, selected_injuries)
    else:
        label.config(text="No image selected")

# Initialize the Tkinter window
root = tk.Tk()
root.title("Face Injury Simulation")

# Label for selecting image
label = tk.Label(root, text="No image selected")
label.pack()

# Button to select image
select_button = tk.Button(root, text="Select Image", command=open_image)
select_button.pack()

# Injury Type Listbox
injury_var = tk.StringVar(value=["bruise", "cut", "burn", "black eye"])
injury_listbox = tk.Listbox(root, listvariable=injury_var, selectmode='multiple')
injury_listbox.pack()

# Apply Injury Button
apply_button = tk.Button(root, text="Apply Injury", command=apply_injury_gui)
apply_button.pack()

root.mainloop()
