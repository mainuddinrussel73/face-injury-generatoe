import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from utils.landmark_detection import get_facial_landmarks


def apply_injury(face_image, injury_type, intensity=1.0):
    # Load injury overlay based on selected injury type
    injury_path = f"images/{injury_type}.png"
    injury_img = Image.open(injury_path).convert("RGBA")

    # Resize the injury overlay with different scaling for the bruise
    if injury_type == 'bruise':
        # Scale down the bruise image more significantly for a realistic look
        scale_factor = 0.15  # 15% of the face width
    elif injury_type == 'black eye':
        # Scale down the black eye image for appropriate placement
        scale_factor = 0.2  # 10% of the face widthburn
    elif injury_type == 'cut':
        # Scale down the black eye image for appropriate placement
        scale_factor = 0.15  # 10% of the face widthburn
    else:
        scale_factor = 0.3  # 30% for other injuries

    injury_resized = injury_img.resize((int(face_image.width * scale_factor), int(face_image.height * scale_factor)))

    # Detect facial landmarks
    landmarks = get_facial_landmarks(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))

    # Place injury at a specific landmark based on type
    if injury_type == 'bruise':
        # Place bruise between the nose and left cheek
        bruise_x = (landmarks[30][0] + landmarks[15][0]) // 2
        bruise_y = landmarks[30][1] + 30  # Adjusted downwards for placement
        injury_position = (bruise_x - injury_resized.width // 2, bruise_y - injury_resized.height // 2)
    elif injury_type == 'cut':
        # Place cut on the forehead between the eyebrows
        forehead_x = (landmarks[21][0] + landmarks[22][0]) // 2  # Midpoint between left (21) and right (22) eyebrows
        forehead_y = landmarks[21][1] - 160  # Adjusted upwards
        injury_position = (forehead_x - injury_resized.width // 2, forehead_y - injury_resized.height // 2)
    elif injury_type == 'black eye':
        # Position black eye on upper left eyelid
        injury_position = (landmarks[39][0] - injury_resized.width // 2, landmarks[39][1] + 10)  # Adjusted downwards
    elif injury_type == 'burn':
        # Position burn on upper right eyelid
        injury_position = (
        landmarks[43][0] - injury_resized.width // 2, landmarks[43][1] - injury_resized.height // 2)  # Adjusted upwards

    # Adjust intensity
    enhancer = ImageEnhance.Brightness(injury_resized)
    injury_resized = enhancer.enhance(intensity)

    # Apply the injury overlay on the face
    face_image.paste(injury_resized, injury_position, injury_resized)

    return face_image  # Return the updated image


def apply_injury_multiple(face_image_path, selected_injuries):
    # Load the face image
    face_image = cv2.imread(face_image_path)
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_image)

    for injury in selected_injuries:
        intensity = float(input(f"Enter the intensity for {injury} (0.0 to 1.0): "))
        if intensity < 0.0 or intensity > 1.0:
            print("Intensity should be between 0.0 and 1.0. Setting to default 0.5.")
            intensity = 0.5
        face_pil = apply_injury(face_pil, injury, intensity)  # Update the face image with new injury

    # Convert back to OpenCV format and save the output
    output_image_path = "images/output/injured_face.jpg"  # Output directory
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, cv2.cvtColor(np.array(face_pil), cv2.COLOR_RGB2BGR))
    print(f"Injured face saved to {output_image_path}")


def main():
    # Input image path
    face_image_path = "C:\\Users\\DoICT\\Downloads\\demo1.webp"

    # Input multiple injury types
    print("Available injuries: bruise, cut, burn, black eye")
    selected_injuries = input("Enter the injuries you want to apply (comma-separated): ").split(",")

    # Strip whitespace from selected injuries
    selected_injuries = [injury.strip() for injury in selected_injuries]

    # Apply injuries
    apply_injury_multiple(face_image_path, selected_injuries)


if __name__ == "__main__":
    main()
