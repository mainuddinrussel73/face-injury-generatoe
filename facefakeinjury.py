import cv2
import dlib
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor_path  =  "C:/Users/DoICT/PycharmProjects/addfaceinjury/utils/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Global variables to hold images and states
img = None
img_display = None
original_img = None

# Function to apply black eye effect
def apply_black_eye(output_img, eye_points, severity, opacity):
    # Get bounding box around the eye
    x, y, w, h = cv2.boundingRect(eye_points)

    # Extract the eye region of interest (ROI)
    eye_roi = output_img[y:y+h, x:x+w]

    # Darken the eye area to simulate the black eye (increase severity of darkness)
    dark_eye = eye_roi.copy()
    dark_eye[:, :, 0] = np.clip(dark_eye[:, :, 0] * (1 - severity / 100), 0, 255)  # Reduce blue channel
    dark_eye[:, :, 1] = np.clip(dark_eye[:, :, 1] * (1 - severity / 100), 0, 255)  # Reduce green channel
    dark_eye[:, :, 2] = cv2.add(dark_eye[:, :, 2], int(severity * 1.5))  # Increase red channel

    # Blend the darkened eye region back onto the face using the opacity
    blended_eye = cv2.addWeighted(eye_roi, 1 - opacity / 100, dark_eye, opacity / 100, 0)
    output_img[y:y+h, x:x+w] = blended_eye

# Global variable to store the modified PIL image for saving
final_pil_image = None

# Function to apply swollen lips and black eye effects
def apply_injury(swelling_intensity, bruising_intensity, black_eye_severity, black_eye_opacity):
    global img, img_display, original_img, final_pil_image

    # Check if the original image is loaded
    if original_img is None:
        messagebox.showerror("Error", "Please load an image first.")
        return

    # Reset to the original image
    output_img = original_img.copy()

    # Convert the image to grayscale
    gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Lip Region (landmarks 48-67 for lips)
        lip_points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(48, 68)])

        # Get bounding box around the lips
        x, y, w, h = cv2.boundingRect(lip_points)

        # Extract the lip region of interest (ROI)
        lips_roi = output_img[y:y+h, x:x+w]

        # Enlarge the lips (swollen effect) based on the slider value
        lips_swollen = cv2.resize(lips_roi, None, fx=1 + swelling_intensity / 100, fy=1 + swelling_intensity / 100, interpolation=cv2.INTER_CUBIC)

        # Adjust lip color to mimic bruising (reddish-purple hue) based on the bruising intensity slider
        lips_swollen[:, :, 1] = lips_swollen[:, :, 1] * (1 - bruising_intensity / 100)  # Reduce green channel
        lips_swollen[:, :, 2] = cv2.add(lips_swollen[:, :, 2], int(bruising_intensity * 2))  # Increase red channel

        # Resize the swollen lips back to the original lip region size
        lips_swollen_resized = cv2.resize(lips_swollen, (w, h), interpolation=cv2.INTER_AREA)

        # Replace the original lip area with the swollen lip region
        output_img[y:y+h, x:x+w] = lips_swollen_resized

        # Apply Black Eye (for left eye, landmarks 36-41)
        left_eye_points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(36, 42)])
        apply_black_eye(output_img, left_eye_points, black_eye_severity, black_eye_opacity)

        # Apply Black Eye (for right eye, landmarks 42-47)
        right_eye_points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(42, 48)])
        apply_black_eye(output_img, right_eye_points, black_eye_severity, black_eye_opacity)

    # Convert the output image to RGB format for displaying in Tkinter
    img_display_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    final_pil_image = Image.fromarray(img_display_rgb)  # Store PIL image for saving

    # Update the Tkinter display with the modified image
    img_display = ImageTk.PhotoImage(final_pil_image)
    canvas.create_image(0, 0, anchor=NW, image=img_display)


# Function to load the input image
def load_image():
    global img, original_img
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        original_img = img.copy()  # Store original image for resetting
        img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_display = Image.fromarray(img_display)
        img_display = ImageTk.PhotoImage(img_display)
        canvas.create_image(0, 0, anchor=NW, image=img_display)


# Function to save the modified image
def save_image():
    global final_pil_image

    if final_pil_image is not None:
        # Ask the user where to save the image
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])

        if save_path:
            # Save the image using PIL
            final_pil_image.save(save_path)
            messagebox.showinfo("Image Saved", f"Image saved to {save_path}")
    else:
        messagebox.showwarning("No Image", "No modified image to save.")


# Function to reset the image to the original state
def reset_image():
    global original_img, img_display
    if original_img is not None:
        img_display = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img_display = Image.fromarray(img_display)
        img_display = ImageTk.PhotoImage(img_display)
        canvas.create_image(0, 0, anchor=NW, image=img_display)

# Tkinter GUI setup
root = Tk()
root.title("Injury Simulation: Swollen Lips and Black Eye")

# Canvas to display the image
canvas = Canvas(root, width=500, height=500)
canvas.pack()

# Button to load the image
load_button = Button(root, text="Load Image", command=load_image)
load_button.pack()

# Button to save the modified image
save_button = Button(root, text="Save Image", command=save_image)
save_button.pack()

# Button to reset the image
reset_button = Button(root, text="Reset Image", command=reset_image)
reset_button.pack()

# Slider to control swelling intensity for lips
swelling_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL, label="Lip Swelling Intensity", command=lambda x: apply_injury(swelling_slider.get(), bruising_slider.get(), black_eye_slider.get(), black_eye_opacity_slider.get()))
swelling_slider.set(20)  # Initial value
swelling_slider.pack()

# Slider to control bruising severity for lips
bruising_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL, label="Lip Bruising Severity", command=lambda x: apply_injury(swelling_slider.get(), bruising_slider.get(), black_eye_slider.get(), black_eye_opacity_slider.get()))
bruising_slider.set(50)  # Initial value
bruising_slider.pack()

# Slider to control black eye severity
black_eye_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL, label="Black Eye Severity", command=lambda x: apply_injury(swelling_slider.get(), bruising_slider.get(), black_eye_slider.get(), black_eye_opacity_slider.get()))
black_eye_slider.set(30)  # Initial value
black_eye_slider.pack()

# Slider to control black eye opacity
black_eye_opacity_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL, label="Black Eye Opacity", command=lambda x: apply_injury(swelling_slider.get(), bruising_slider.get(), black_eye_slider.get(), black_eye_opacity_slider.get()))
black_eye_opacity_slider.set(50)  # Initial value
black_eye_opacity_slider.pack()

# Start the Tkinter loop
root.mainloop()
