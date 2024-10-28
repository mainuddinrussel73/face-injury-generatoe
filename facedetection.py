import dlib
import cv2
import os
import numpy as np


# Load burn images (e.g., burn1.png and burn2.png with transparency)
burn_images = [
    cv2.imread("C:/Users/DoICT/PycharmProjects/addfaceinjury/images/burn.png", cv2.IMREAD_UNCHANGED),  # Burn effect 1
    cv2.imread("C:/Users/DoICT/PycharmProjects/addfaceinjury/images/burn.png", cv2.IMREAD_UNCHANGED)   # Burn effect 2
]

# Load the injury image (e.g., bruise.png with transparency)
injury_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/cut.png"
injury_image = cv2.imread(injury_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel



# Load blood drop image (e.g., blood_drop.png with transparency)
blood_drop_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/—Pngtree—vector wound bleeding blood drop_5771274.png"
blood_drop_image = cv2.imread(blood_drop_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel


# Path to the landmark predictor and image
predictor_path  =  "C:/Users/DoICT/PycharmProjects/addfaceinjury/utils/shape_predictor_68_face_landmarks.dat"
image_path = "C:/Users/DoICT/Downloads/face2.jpg" #"C:/Users/DoICT/PycharmProjects/addfaceinjury/images/face_image.jpg"

# Function to apply a blood drop to the image
def apply_blood_drop(image, blood_drop_image, position, scale=1.0):
    # Resize the blood drop image
    blood_drop_image = cv2.resize(blood_drop_image, (int(blood_drop_image.shape[1] * scale), int(blood_drop_image.shape[0] * scale)))

    # Extract the alpha channel from the blood drop image
    blood_drop_alpha = blood_drop_image[:, :, 3] / 255.0  # Alpha channel
    blood_drop_rgb = blood_drop_image[:, :, :3]  # RGB channels

    # Get position for the blood drop
    x_offset, y_offset = position
    h, w = blood_drop_image.shape[:2]

    # Check if the blood drop fits within the original image
    if y_offset + h > image.shape[0] or x_offset + w > image.shape[1]:
        print("Blood drop image exceeds original image bounds. Adjusting position.")
        return image

    # Get the region of interest (ROI) from the main image
    roi = image[y_offset:y_offset + h, x_offset:x_offset + w]

    # Blend the blood drop image with the ROI using the alpha channel
    for c in range(3):  # Loop over the color channels
        roi[:, :, c] = (blood_drop_rgb[:, :, c] * blood_drop_alpha + roi[:, :, c] * (1 - blood_drop_alpha))

    # Place the blended blood drop back on the main image
    image[y_offset:y_offset + h, x_offset:x_offset + w] = roi

    return image


# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: The image file at {image_path} does not exist.")
else:
    # Load the face detector and the shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Load the input image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not loaded. Check the file path or file format.")
    else:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)

        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)

            # Dictionary to hold face parts and their corresponding landmarks
            face_parts = {
                "Jaw": range(0, 17),
                "Right Eyebrow": range(17, 22),
                "Left Eyebrow": range(22, 27),
                "Nose": range(27, 36),
                "Right Eye": range(36, 42),
                "Left Eye": range(42, 48),
                "Mouth Outer": range(48, 60),
                "Mouth Inner": range(60, 68),
            }

            # Draw the locations of landmarks (larger, more intense dots)
            for part, indices in face_parts.items():
                for i in indices:
                    x, y = landmarks.part(i).x, landmarks.part(i).y

                    # Draw larger circles with thicker borders
                    cv2.circle(image, (x, y), 4, (0, 255, 0), -1)  # Larger green circles


            ### Estimating Forehead and Cheeks ###


            # Forehead: Estimate it based on the upper part of the face above eyebrows
            # Get mid-point between eyebrows
            forehead_x = (landmarks.part(19).x + landmarks.part(24).x) // 2
            forehead_y = (landmarks.part(19).y + landmarks.part(24).y) // 2 - 200  # Increase the offset to move up
            cv2.circle(image, (forehead_x, forehead_y), 6, (255, 0, 0), -1)  # Blue dot for forehead

            # Cheeks: Estimating using area below eyes and beside the nose
            # Right cheek: Below right eye and right of nose
            right_cheek_x = landmarks.part(36).x + 20  # A bit to the right of the right eye
            right_cheek_y = landmarks.part(29).y + 20  # A bit below the nose
            cv2.circle(image, (right_cheek_x, right_cheek_y), 6, (0, 0, 255), -1)  # Red dot for right cheek

            # Left cheek: Below left eye and left of nose
            left_cheek_x = landmarks.part(45).x - 20  # A bit to the left of the left eye
            left_cheek_y = landmarks.part(29).y + 20  # A bit below the nose
            cv2.circle(image, (left_cheek_x, left_cheek_y), 6, (0, 0, 255), -1)  # Red dot for left cheek

            ### Calculate Forehead Size ###
            # Use the distance between the eyebrows (width) and the vertical distance from eyes to top of the forehead (height)
            eyebrow_distance = abs(landmarks.part(24).x - landmarks.part(19).x)  # Width of forehead
            forehead_height = abs(landmarks.part(19).y - forehead_y) + 50  # Height estimation

            # Calculate the injury size as 30% of the forehead area
            injury_width = int(eyebrow_distance * 0.5)  # 30% of the forehead width
            injury_height = int(forehead_height * 0.5)  # 30% of the forehead height

            ### Apply Injury to the Forehead ###
            # Resize the injury image to fit the calculated size
            resized_injury = cv2.resize(injury_image, (injury_width, injury_height))

            # Calculate the region of interest (ROI) on the forehead
            x_start = forehead_x - injury_width // 2
            y_start = forehead_y - injury_height // 2
            x_end = x_start + injury_width
            y_end = y_start + injury_height

            # Check if ROI dimensions are valid
            if y_start < 0 or x_start < 0 or y_end > image.shape[0] or x_end > image.shape[1]:
                print("Error: ROI for forehead goes out of image bounds.")
                print(f"ROI Coordinates: (x_start: {x_start}, y_start: {y_start}, x_end: {x_end}, y_end: {y_end})")
            else:
                # Extract the alpha channel from the injury image
                injury_alpha = resized_injury[:, :, 3] / 255.0  # Alpha channel (transparency)
                injury_rgb = resized_injury[:, :, :3]  # RGB channels

                # Get the region of interest (ROI) from the main image where injury will be applied
                forehead_roi = image[y_start:y_end, x_start:x_end]

                # Print shapes for debugging
                print(f"Forehead ROI shape: {forehead_roi.shape}")
                print(f"Resized Injury shape: {resized_injury.shape}")

                # Check if forehead_roi is empty
                if forehead_roi.size == 0:
                    print("Error: Forehead ROI is empty. Please check the coordinates.")
                else:
                    # Blend the injury image with the forehead region using the alpha channel
                    for c in range(3):  # Loop over the color channels (B, G, R)
                        forehead_roi[:, :, c] = (injury_rgb[:, :, c] * injury_alpha +
                                                 forehead_roi[:, :, c] * (1.0 - injury_alpha))

                    # Place the blended injury back on the main image
                    image[y_start:y_end, x_start:x_end] = forehead_roi

            ### Apply Burns Only on the Right Eye ###
            # Get right eye landmarks
            right_eye_indices = range(36, 42)

            # Calculate the position of the right eye
            right_eye_x = sum(landmarks.part(i).x for i in right_eye_indices) // len(right_eye_indices)-100
            right_eye_y = sum(landmarks.part(i).y for i in right_eye_indices) // len(right_eye_indices)

            # Apply burn effects around the right eye
            for j, burn_image in enumerate(burn_images):
                # Resize burn image to fit the area around the right eye
                burn_width = int(right_eye_x * 0.5)  # Adjust size based on the eye position
                burn_height = int(right_eye_y * 0.5)  # Adjust size based on the eye position
                resized_burn = cv2.resize(burn_image, (burn_width, burn_height))

                # Calculate the position to place the burn image
                x_start = right_eye_x - (burn_width // 2)
                y_start = right_eye_y - (burn_height // 2) - 10 * (j + 1)  # Offset for different burns
                x_end = x_start + burn_width
                y_end = y_start + burn_height

                # Check if the burn image fits within the image boundaries
                if x_start < 0 or y_start < 0 or x_end > image.shape[1] or y_end > image.shape[0]:
                    continue

                # Extract the alpha channel from the burn image
                burn_alpha = resized_burn[:, :, 3] / 255.0  # Alpha channel (transparency)
                burn_rgb = resized_burn[:, :, :3]  # RGB channels

                # Get the region of interest (ROI) from the main image
                eye_roi = image[y_start:y_end, x_start:x_end]

                # Blend the burn image with the eye region using the alpha channel
                for c in range(3):  # Loop over the color channels (B, G, R)
                    eye_roi[:, :, c] = (burn_rgb[:, :, c] * burn_alpha +
                                        eye_roi[:, :, c] * (1.0 - burn_alpha))

                # Place the blended burn back on the main image
                image[y_start:y_end, x_start:x_end] = eye_roi

            ### Apply Blood Drop on the Left Cheek ###
            blood_drop_position = (left_cheek_x - 185, left_cheek_y - 20)  # Adjust position for left cheek
            image = apply_blood_drop(image, blood_drop_image, blood_drop_position,
                                         scale=1)  # Scale can be adjusted

        # Save the image with landmarks marked
        output_image_path = "output_image_with_intense_landmarks.jpg"
        cv2.imwrite(output_image_path, image)

        # Display the output image (optional)
        cv2.imshow("Facial landmarks", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"Output image saved as {output_image_path}")
