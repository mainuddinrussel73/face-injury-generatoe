import dlib
import cv2
import os
import numpy as np

# Load the pre-trained face detector and shape predictor
predictor_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/utils/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Load the injury images (with transparency)
scar_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/blood_drop.png"
scar_image = cv2.imread(scar_image_path, cv2.IMREAD_UNCHANGED)

burn_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/burn.png"
burn_image = cv2.imread(burn_image_path, cv2.IMREAD_UNCHANGED)

cut_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/cut.png"
cut_image = cv2.imread(cut_image_path, cv2.IMREAD_UNCHANGED)

swollen_eye_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/wound-png-47525.png"
swollen_eye_image = cv2.imread(swollen_eye_path, cv2.IMREAD_UNCHANGED)


def load_image(image_path):
    """Load an image from the given path."""
    return cv2.imread(image_path)


def detect_faces(image):
    """Detect faces in the input image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return faces


def get_landmarks(image, face):
    """Get facial landmarks for the given face region."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return predictor(gray, face)


def resize_injury(image, width, height):
    """Resize injury image to the given width and height."""
    return cv2.resize(image, (width, height))


def blend_images(background, overlay, position, alpha_channel):
    """Blend the overlay image with the background at a given position."""
    x_start, y_start, x_end, y_end = position
    for c in range(3):  # Loop over the color channels (B, G, R)
        background[y_start:y_end, x_start:x_end, c] = (
            overlay[:, :, c] * alpha_channel + background[y_start:y_end, x_start:x_end, c] * (1.0 - alpha_channel)
        )


def apply_scar(image, landmarks, scar_image):
    """Apply a scar image on the top left of the forehead."""
    left_eyebrow_x = landmarks.part(19).x
    left_eyebrow_y = landmarks.part(19).y

    # Calculate forehead position for scar
    forehead_x = left_eyebrow_x - int(left_eyebrow_x * 0.1)
    forehead_y = left_eyebrow_y - int(left_eyebrow_x * 0.1)

    face_width = landmarks.part(16).x - landmarks.part(0).x
    scar_width = int(face_width * 0.2)
    scar_height = int(scar_width * 0.5)

    resized_scar = resize_injury(scar_image, scar_width, scar_height)

    # Define the region for the scar
    x_start = forehead_x - scar_width // 2
    y_start = forehead_y - scar_height // 2
    x_end = x_start + scar_width
    y_end = y_start + scar_height

    if x_start < 0 or y_start < 0 or x_end > image.shape[1] or y_end > image.shape[0]:
        return

    scar_alpha = resized_scar[:, :, 3] / 255.0  # Alpha channel
    scar_rgb = resized_scar[:, :, :3]  # RGB channels

    blend_images(image, scar_rgb, (x_start, y_start, x_end, y_end), scar_alpha)


def process_image(image_path, output_dir):
    """Process an image, detect facial landmarks, and apply injuries."""
    image = load_image(image_path)
    faces = detect_faces(image)

    for face in faces:
        landmarks = get_landmarks(image, face)

        # Apply different injuries
        apply_scar(image, landmarks, swollen_eye_image)

        # Save the processed image
        output_filename = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_filename, image)
        print(f"Processed image saved as: {output_filename}")


def process_directory(input_dir, output_dir):
    """Process all images in the input directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")
            process_image(image_path, output_dir)


# Example usage
input_directory = "C:/Users/DoICT/PycharmProjects/addfaceinjury/input_images/"
output_directory = "C:/Users/DoICT/PycharmProjects/addfaceinjury/output_images/"
process_directory(input_directory, output_directory)
