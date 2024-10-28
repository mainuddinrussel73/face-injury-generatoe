import dlib
import cv2
import os
import numpy as np
import random
from mtcnn import MTCNN


# Load the pre-trained face detector and shape predictor
predictor_path  =  "C:/Users/DoICT/PycharmProjects/addfaceinjury/utils/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


# Load blood drop image (e.g., blood_drop.png with transparency)
blood_drop_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/—Pngtree—vector wound bleeding blood drop_5771274.png"
blood_drop_image = cv2.imread(blood_drop_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel
# Load the injury image (e.g., bruise.png with transparency)
scar_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/—Pngtree—surgical scars and bruises_6843543.png"
scar_image = cv2.imread(scar_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

# Load the injury image (e.g., bruise.png with transparency)
stitch_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/wound-png-47525.png"
stitch_image = cv2.imread(stitch_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel


burn_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/burn.png"  # Update this path
burn_image = cv2.imread(burn_image_path, cv2.IMREAD_UNCHANGED)

cut_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/cut.png"
cut_image = cv2.imread(cut_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel


# Load the swollen eye image (PNG with transparency)
swollen_eye_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/pngegg (3).png"
swollen_eye_image = cv2.imread(swollen_eye_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

# Load the swollen eye image (PNG with transparency)
swollens_eye_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/eye_swollen.png"
swollens_eye_image = cv2.imread(swollens_eye_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel


bloods_drop_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/bloods.png"
bloods_drop_image = cv2.imread(bloods_drop_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

blood_drip_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/blood-drip.png"
blood_drip_image = cv2.imread(blood_drip_path, cv2.IMREAD_UNCHANGED)

wound_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/wound_scar.png"
wound_image = cv2.imread(wound_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

bullet_scar_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/pngegg (2).png"
bullet_scar_image = cv2.imread(bullet_scar_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

punch_mark_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/wound-png-47525.png"
punch_mark_image = cv2.imread(punch_mark_image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel

mark_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/wound-png-47537.png"
mark_image = cv2.imread(mark_image_path, cv2.IMREAD_UNCHANGED)

blister_mark_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/swollen_eyeee.png"
blister_mark_image = cv2.imread(blister_mark_image_path, cv2.IMREAD_UNCHANGED)

slap_mark_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/handprint.png"
slap_mark_image = cv2.imread(slap_mark_image_path, cv2.IMREAD_UNCHANGED)

blackmark_mark_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/bulletscar.png"
blackmark_mark_image = cv2.imread(blackmark_mark_image_path, cv2.IMREAD_UNCHANGED)

bite_mark_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/bite.png"
bite_mark_image = cv2.imread(bite_mark_image_path, cv2.IMREAD_UNCHANGED)

stab_mark_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/stabbedknife.png"
stab_mark_image = cv2.imread(stab_mark_image_path, cv2.IMREAD_UNCHANGED)

patch_mark_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/patchmark.png"
patch_mark_image = cv2.imread(patch_mark_image_path, cv2.IMREAD_UNCHANGED)

blacks_mark_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/swollen_eye222.png"
blacks_mark_image = cv2.imread(blacks_mark_image_path, cv2.IMREAD_UNCHANGED)

big_cut_mark_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/bigcutt.png"
big_cut_mark_image = cv2.imread(big_cut_mark_image_path, cv2.IMREAD_UNCHANGED)


big_eye_mark_image_path = "C:/Users/DoICT/PycharmProjects/addfaceinjury/images/scaryeye.png"
big_eye_mark_image = cv2.imread(big_eye_mark_image_path, cv2.IMREAD_UNCHANGED)

#left cheek
def apply_blood_left_cheek_drop(image, landmarks ,blood_drop_image):
    # Resize the blood drop image
    # Calculate face dimensions using facial landmarks
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between the leftmost and rightmost points
    face_height = landmarks.part(8).y - landmarks.part(19).y  # Rough estimate for face height

    # Define the size of the scar as a proportion of face dimensions
    scar_width = int(face_width * 0.4)  # 20% of face width
    scar_height = int(face_height * 0.25)  # 15% of face height

    # Validate the dimensions
    if scar_width <= 0 or scar_height <= 0:
        print("Error: Invalid scar dimensions.")
        return image

    # Resize the scar image to fit the face size
    resized_scar = cv2.resize(blood_drop_image, (scar_width, scar_height))

    # Define position for the right cheek
    # Example: Use landmark point 13 or 14, which is near the right cheek/jaw area
    cheek_landmark = landmarks.part(13)  # Use part(14) for middle of the right cheek
    left_cheek_x = landmarks.part(45).x  # A bit to the left of the left eye
    left_cheek_y = landmarks.part(29).y  # A bit below the nose

    # Subtract half the scar width/height to center it on the cheek
    scar_x = left_cheek_x - scar_width // 2
    scar_y = left_cheek_y - scar_height // 2
    lift_offset = int(face_height * 0.1)  # Adjust this value to control how much higher to place the stitch
    scar_y += lift_offset
    # Ensure the coordinates are within the image boundaries
    if scar_x + scar_width > image.shape[1]:
        scar_x = image.shape[1] - scar_width
    if scar_y + scar_height > image.shape[0]:
        scar_y = image.shape[0] - scar_height
    if scar_x < 0:
        scar_x = 0
    if scar_y < 0:
        scar_y = 0

    # Add the scar image onto the face (simple paste for now; alpha blending can be added)
    overlay_image(image, resized_scar, scar_x, scar_y)

    return image

#rigth cheek blood
def apply_bullet_right_cheek(image, landmarks, scar_image):
    # Calculate face dimensions using facial landmarks
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between the leftmost and rightmost points
    face_height = landmarks.part(8).y - landmarks.part(19).y  # Rough estimate for face height

    # Define the size of the scar as a proportion of face dimensions
    scar_width = int(face_width * 0.2)  # 20% of face width
    scar_height = int(face_height * 0.5)  # 15% of face height

    # Validate the dimensions
    if scar_width <= 0 or scar_height <= 0:
        print("Error: Invalid scar dimensions.")
        return image

    # Resize the scar image to fit the face size
    resized_scar = cv2.resize(scar_image, (scar_width, scar_height))

    # Define position for the right cheek
    # Example: Use landmark point 13 or 14, which is near the right cheek/jaw area
    cheek_landmark = landmarks.part(13)  # Use part(14) for middle of the right cheek
    left_cheek_x = landmarks.part(45).x   # A bit to the left of the left eye
    left_cheek_y = landmarks.part(29).y   # A bit below the nose

    # Subtract half the scar width/height to center it on the cheek
    scar_x = left_cheek_x - scar_width // 2
    scar_y = left_cheek_y - scar_height // 2



    # Ensure the coordinates are within the image boundaries
    if scar_x + scar_width > image.shape[1]:
        scar_x = image.shape[1] - scar_width
    if scar_y + scar_height > image.shape[0]:
        scar_y = image.shape[0] - scar_height
    if scar_x < 0:
        scar_x = 0
    if scar_y < 0:
        scar_y = 0

    # Add the scar image onto the face (simple paste for now; alpha blending can be added)
    overlay_image(image, resized_scar, scar_x, scar_y)

    return image

def apply_blood_drip_right_cheek(image, landmarks, blood_drip_image):
    # Calculate face dimensions using facial landmarks
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between the leftmost and rightmost points
    face_height = landmarks.part(8).y - landmarks.part(19).y  # Rough estimate for face height

    # Define the size of the scar as a proportion of face dimensions
    scar_width = int(face_width * 0.2)  # 20% of face width
    scar_height = int(face_height * 0.5)  # 15% of face height

    # Validate the dimensions
    if scar_width <= 0 or scar_height <= 0:
        print("Error: Invalid scar dimensions.")
        return image

    # Resize the scar image to fit the face size
    resized_scar = cv2.resize(blood_drip_image, (scar_width, scar_height))

    # Define position for the right cheek
    # Example: Use landmark point 13 or 14, which is near the right cheek/jaw area
    cheek_landmark = landmarks.part(14)  # Use part(14) for middle of the right cheek
    left_cheek_x = landmarks.part(47).x   # A bit to the left of the left eye
    left_cheek_y = landmarks.part(36).y   # A bit below the nose

    # Subtract half the scar width/height to center it on the cheek
    scar_x = left_cheek_x - scar_width // 2
    scar_y = left_cheek_y - scar_height // 2



    # Ensure the coordinates are within the image boundaries
    if scar_x + scar_width > image.shape[1]:
        scar_x = image.shape[1] - scar_width
    if scar_y + scar_height > image.shape[0]:
        scar_y = image.shape[0] - scar_height
    if scar_x < 0:
        scar_x = 0
    if scar_y < 0:
        scar_y = 0

    # Add the scar image onto the face (simple paste for now; alpha blending can be added)
    overlay_image(image, resized_scar, scar_x, scar_y)

    return image



#rigth cheek mark
def apply_mark_right_cheek(image, landmarks, mark_image):
    # Calculate face dimensions using facial landmarks
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between the leftmost and rightmost points
    face_height = landmarks.part(8).y - landmarks.part(19).y  # Rough estimate for face height

    # Define the size of the scar as a proportion of face dimensions
    scar_width = int(face_width * 0.2)  # 20% of face width
    scar_height = int(face_height * 0.5)  # 15% of face height

    # Validate the dimensions
    if scar_width <= 0 or scar_height <= 0:
        print("Error: Invalid scar dimensions.")
        return image

    # Resize the scar image to fit the face size
    resized_scar = cv2.resize(mark_image, (scar_width, scar_height))

    # Define position for the right cheek
    # Example: Use landmark point 13 or 14, which is near the right cheek/jaw area
    cheek_landmark = landmarks.part(13)  # Use part(14) for middle of the right cheek
    left_cheek_x = landmarks.part(45).x   # A bit to the left of the left eye
    left_cheek_y = (landmarks.part(29).y )  # A bit below the nose

    # Subtract half the scar width/height to center it on the cheek
    scar_x = left_cheek_x - scar_width // 2
    scar_y = (left_cheek_y - scar_height // 2)



    # Ensure the coordinates are within the image boundaries
    if scar_x + scar_width > image.shape[1]:
        scar_x = image.shape[1] - scar_width
    if scar_y + scar_height > image.shape[0]:
        scar_y = image.shape[0] - scar_height
    if scar_x < 0:
        scar_x = 0
    if scar_y < 0:
        scar_y = 0

    # Add the scar image onto the face (simple paste for now; alpha blending can be added)
    overlay_image(image, resized_scar, scar_x, scar_y)

    return image


#forehead stitch mark
def apply_stitch_forehead(image, face , landmarks, swollen_eye_image):
    # Calculate face dimensions using facial landmarks
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between the leftmost and rightmost points
    face_height = landmarks.part(8).y - landmarks.part(19).y  # Rough estimate for face height

    # Define the size of the stitch mark as a proportion of face dimensions
    stitch_width = int(face_width * 0.15)  # 15% of face width
    stitch_height = int(face_height * 0.1)  # 10% of face height

    # Validate the dimensions
    if stitch_width <= 0 or stitch_height <= 0:
        print("Error: Invalid stitch mark dimensions.")
        return image

    # Resize the stitch mark image to fit on the forehead
    resized_stitch = cv2.resize(swollen_eye_image, (stitch_width, stitch_height))

    # Define position for the right side of the forehead using landmarks
    # Landmark points around 19, 20, and 24 usually define the forehead area
    forehead_x = landmarks.part(24).x  # Near the right side of the forehead
    forehead_y = landmarks.part(19).y - int(face_height * 0.2)  # A bit higher up for the forehead

    # Subtract half the stitch width/height to center it on the forehead
    stitch_x = forehead_x - stitch_width // 2
    stitch_y = forehead_y - stitch_height // 2

    # Ensure the coordinates are within the image boundaries
    if stitch_x + stitch_width > image.shape[1]:
        stitch_x = image.shape[1] - stitch_width
    if stitch_y + stitch_height > image.shape[0]:
        stitch_y = image.shape[0] - stitch_height
    if stitch_x < 0:
        stitch_x = 0
    if stitch_y < 0:
        stitch_y = 0

    # Add the stitch mark image onto the forehead (simple paste for now; alpha blending can be added)
    overlay_image(image, resized_stitch, stitch_x, stitch_y)

    return image


#burn mark
def apply_burn_mark(image , landmarks,burn_image ): ### Apply Burns Below the Right Eye ###
        # Right eye landmarks
        right_eye_indices = range(36, 42)

        # Calculate the position of the right eye
        right_eye_x = sum(landmarks.part(i).x for i in right_eye_indices) // len(right_eye_indices)
        right_eye_y = sum(landmarks.part(i).y for i in right_eye_indices) // len(right_eye_indices)

        # Calculate the width and height of the eye for burn size
        eye_width = landmarks.part(39).x - landmarks.part(36).x  # Width of the right eye
        eye_height = landmarks.part(41).y - landmarks.part(37).y  # Height of the right eye

        # Define the size of the burn effect based on the eye size
        burn_width = int(eye_width * 2)  # Burn width is now 2x the eye width
        burn_height = int(eye_height * 2)  # Burn height is now 2x the eye height

        # Resize burn image to fit the area around the right eye
        resized_burn = cv2.resize(burn_image, (burn_width, burn_height))

        # Define the position for the burn image to be below the right eye
        x_start = right_eye_x - (burn_width // 2)  # Center burn horizontally
        y_start = right_eye_y + eye_height + 5  # Place burn below the eye with a slight vertical offset

        # Define the bottom-right corner of the burn
        x_end = x_start + burn_width
        y_end = y_start + burn_height


        # Check if the burn image fits within the image boundaries
        if x_start < 0 or y_start < 0 or x_end > image.shape[1] or y_end > image.shape[0]:
            return image


        overlay_image(image, resized_burn, x_start, y_start)

        return image

#cut mark left cheek
def apply_cut_left_cheek(image, landmarks, cut_image):
    # Calculate the position for the scar between the nose and left cheek
    nose_x = landmarks.part(30).x
    nose_y = landmarks.part(30).y
    left_cheek_x = landmarks.part(1).x
    left_cheek_y = landmarks.part(1).y

    # Position scar in between nose and left cheek
    scar_x = (nose_x + left_cheek_x) // 2
    scar_y = (nose_y + (left_cheek_y)) // 2

    # Calculate the width of the face to determine scar size
    face_width = abs(landmarks.part(16).x - landmarks.part(0).x)  # Distance between jaw landmarks
    scar_width = int(face_width * .3)  # Scar width is 10% of the face width
    scar_height = int(scar_width * .5)  # Adjust height relative to width (25% of width)

    # Resize the scar image to fit appropriately
    resized_scar = cv2.resize(cut_image, (scar_width, scar_height))

    # Define the region of interest (ROI) for the scar placement
    x_start = scar_x - scar_width // 2
    y_start = scar_y - scar_height // 2
    x_end = x_start + scar_width
    y_end = y_start + scar_height

    # Ensure the ROI is within image boundaries
    if x_start < 0 or y_start < 0 or x_end > image.shape[1] or y_end > image.shape[0]:
        return image

    # Extract the alpha channel from the scar image
    scar_alpha = resized_scar[:, :, 3] / 255.0  # Alpha channel (transparency)
    scar_rgb = resized_scar[:, :, :3]  # RGB channels

    # Get the region of interest (ROI) from the main image
    scar_roi = image[y_start:y_end, x_start:x_end]

    # Blend the scar image with the ROI using the alpha channel
    for c in range(3):  # Loop over the color channels (B, G, R)
        scar_roi[:, :, c] = (scar_rgb[:, :, c] * scar_alpha +
                             scar_roi[:, :, c] * (1.0 - scar_alpha))

    # Place the blended scar back on the main image
    image[y_start:y_end, x_start:x_end] = scar_roi
    # Define cut size based on face width
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between jaw landmarks
    cut_width = int(face_width * 0.3)  # Cut width is 30% of the face width
    cut_height = int(cut_width * 0.2)  # Cut height is 20% of the cut width

    forehead_y = (landmarks.part(19).y + landmarks.part(24).y) // 2
    # Chin (landmark 8)
    chin_y = landmarks.part(8).y

    # Face height calculation
    face_height = chin_y - forehead_y
    ### Apply Cut on the Forehead ###
    # Get mid-point between eyebrows for forehead position
    forehead_x = (landmarks.part(19).x + landmarks.part(24).x) // 2
    forehead_y = (landmarks.part(19).y + landmarks.part(24).y) // 2 - int(face_height * 0.1)  # Move up for cut

    # Resize cut image to fit the calculated size
    resized_cut = cv2.resize(cut_image, (cut_width, cut_height))

    # Calculate position for the cut on the forehead
    x_start_cut = forehead_x - cut_width // 2  # Center cut horizontally
    y_start_cut = forehead_y - cut_height // 2  # Center cut vertically

    # Check if the cut image fits within the image boundaries
    if x_start_cut < 0 or y_start_cut < 0 or (x_start_cut + cut_width) > image.shape[1] or (
            y_start_cut + cut_height) > image.shape[0]:
        return  image

    overlay_image(image, resized_cut, x_start, y_start)

    return image

#wound mark
def apply_wound_right_forehead(image, landmarks, stitch_image):
    # Calculate face dimensions using facial landmarks
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between leftmost and rightmost points
    face_height = landmarks.part(8).y - landmarks.part(19).y  # Rough estimate for face height

    # Define the size of the stitch mark as a smaller proportion of face dimensions
    stitch_width = int(face_width * 0.4)  # 10% of face width
    stitch_height = int(face_height * 0.2)  # 8% of face height

    # Validate the dimensions
    if stitch_width <= 0 or stitch_height <= 0:
        print("Error: Invalid stitch dimensions.")
        return image

    # Resize the stitch mark image to fit the face size
    resized_stitch = cv2.resize(stitch_image, (stitch_width, stitch_height))

    # Define position for the right side of the forehead using landmarks
    # Use landmark part(24) for the right side of the forehead (near the right eyebrow)
    forehead_landmark = landmarks.part(24)  # Right side of the forehead

    # Calculate the stitch mark position
    stitch_x = forehead_landmark.x - stitch_width // 2
    stitch_y = forehead_landmark.y - stitch_height // 2

    # Lift the stitch mark slightly upward by subtracting an offset (e.g., 10 pixels)
    lift_offset = int(face_height * 0.1)  # Adjust this value to control how much higher to place the stitch
    stitch_y -= lift_offset

    # Ensure the coordinates are within the image boundaries
    if stitch_x + stitch_width > image.shape[1]:
        stitch_x = image.shape[1] - stitch_width
    if stitch_y + stitch_height > image.shape[0]:
        stitch_y = image.shape[0] - stitch_height
    if stitch_x < 0:
        stitch_x = 0
    if stitch_y < 0:
        stitch_y = 0

    # Add the stitch mark onto the forehead
    overlay_image(image, resized_stitch, stitch_x, stitch_y)

    return image
#cut mark between eyebrow
def apply_cut_between_eyebrow(image, landmarks, cut_image):
    # Calculate face dimensions using facial landmarks
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between leftmost and rightmost points
    face_height = landmarks.part(8).y - landmarks.part(19).y  # Rough estimate for face height

    # Define the size of the cut image as a smaller proportion of face dimensions
    cut_width = int(face_width * 0.15)  # 15% of face width for the cut size
    cut_height = int(face_height * 0.08)  # 8% of face height for the cut size

    # Resize the cut image to fit the forehead size
    resized_cut = cv2.resize(cut_image, (cut_width, cut_height))

    # Define the position for the cut mark on the center of the forehead
    # Use landmarks part(27) for the middle of the forehead (between the eyes)
    forehead_x = (landmarks.part(19).x + landmarks.part(24).x) // 2
    forehead_y = landmarks.part(19).y - 0  # Adjust as needed

    # Calculate the cut's position
    cut_x = forehead_x - cut_width // 2
    cut_y = forehead_y - cut_height // 2

    # Lift the cut slightly upward if necessary
    lift_offset = (50 ) # Adjust this value to control how much higher to place the cut

    # Ensure the coordinates are within the image boundaries
    if cut_x + cut_width > image.shape[1]:
        cut_x = image.shape[1] - cut_width
    if cut_y + cut_height > image.shape[0]:
        cut_y = image.shape[0] - cut_height
    if cut_x < 0:
        cut_x = 0
    if cut_y < 0:
        cut_y = 0

    # Add the cut mark onto the forehead
    overlay_image(image, resized_cut, cut_x, cut_y)

    return image

#cut mark between eyebrow
def apply_big_cut_between_eyebrow(image, landmarks, big_cut_mark_image):
    # Calculate face dimensions using facial landmarks
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between leftmost and rightmost points
    face_height = landmarks.part(8).y - landmarks.part(19).y  # Rough estimate for face height

    # Define the size of the cut image as a smaller proportion of face dimensions
    cut_width = int(face_width * 0.45)  # 15% of face width for the cut size
    cut_height = int(face_height * 0.26)  # 8% of face height for the cut size

    # Resize the cut image to fit the forehead size
    resized_cut = cv2.resize(big_cut_mark_image, (cut_width, cut_height))

    # Define the position for the cut mark on the center of the forehead
    # Use landmarks part(27) for the middle of the forehead (between the eyes)
    forehead_x = (landmarks.part(19).x + landmarks.part(24).x) // 2
    forehead_y = landmarks.part(19).y - 0  # Adjust as needed

    # Calculate the cut's position
    cut_x = forehead_x - cut_width // 2
    cut_y = forehead_y - cut_height // 2

    # Lift the cut slightly upward if necessary
    lift_offset = (50 ) # Adjust this value to control how much higher to place the cut

    # Ensure the coordinates are within the image boundaries
    if cut_x + cut_width > image.shape[1]:
        cut_x = image.shape[1] - cut_width
    if cut_y + cut_height > image.shape[0]:
        cut_y = image.shape[0] - cut_height
    if cut_x < 0:
        cut_x = 0
    if cut_y < 0:
        cut_y = 0

    # Add the cut mark onto the forehead
    overlay_image(image, resized_cut, cut_x, cut_y)

    return image

def apply_red_eye(image, landmarks):
    # Get the left and right eye landmarks
    left_eye_points = [36, 37, 38, 39, 40, 41]  # Left eye landmark indices
    right_eye_points = [42, 43, 44, 45, 46, 47]  # Right eye landmark indices

    # Draw red tint on both eyes
    image = apply_eye_tint(image, landmarks, left_eye_points, (0, 0, 255))  # Red color
    image = apply_eye_tint(image, landmarks, right_eye_points, (0, 0, 255))  # Red color

    return image


def apply_eye_tint(image, landmarks, eye_points, color):
    mask = np.zeros_like(image)

    # Get eye region based on landmarks
    eye_region = np.array([(landmarks.part(p).x, landmarks.part(p).y) for p in eye_points], np.int32)

    # Fill the eye region with red tint
    cv2.fillPoly(mask, [eye_region], color)

    # Blend the eye tint with the original image
    alpha = 0.4  # Transparency factor
    image = cv2.addWeighted(image, 1.0, mask, alpha, 0)

    return image


def apply_red_lips(image, landmarks):
    # Lip landmark points
    lip_points = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]  # Outer lip

    # Get the lip region
    lip_region = np.array([(landmarks.part(p).x, landmarks.part(p).y) for p in lip_points], np.int32)

    # Create a mask for the lip region
    mask = np.zeros_like(image)

    # Fill the lip region with red color (BGR format: Blue, Green, Red)
    cv2.fillPoly(mask, [lip_region], (255, 0, 255))  # Red tint

    # Blend the red lips with the original image
    alpha = 0.6  # Transparency factor for redness
    image = cv2.addWeighted(image, 1.0, mask, alpha, 0)

    return image


def apply_swollen_lips(image, landmarks):
    # Lip landmark points
    lip_points = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]  # Outer lip

    # Get the lip region
    lip_region = np.array([(landmarks.part(p).x, landmarks.part(p).y) for p in lip_points], np.int32)

    # Calculate bounding box for the lips
    x, y, w, h = cv2.boundingRect(lip_region)

    # Extract the lip region from the image
    lips = image[y:y + h, x:x + w]

    # Scale the lips for swollen effect (enlarge by 10%)
    swollen_lips = cv2.resize(lips, None, fx=1.1, fy=1.1)

    # Replace the lips on the original image
    image[y:y + swollen_lips.shape[0], x:x + swollen_lips.shape[1]] = swollen_lips

    # Apply redness to the lips

    return image


def add_blood_drop_image(image, face, landmarks, bloods_drop_image):
    # blood_drop is already a NumPy array (the loaded image), no need to load again

    # Get lip bottom center point (landmark 57)
    right_nostril = (landmarks.part(32).x, landmarks.part(32).y)

    face_width = face.right() - face.left()  # Estimate face width
    blood_width = int(face_width * 0.4)  # 20% of face width for scar
    blood_height = int(blood_width * 0.7)  # Adjust the height of the scar

    # Resize the scar image to fit the forehead
    # Resize blood drop image if necessary
    blood_drop_resized = cv2.resize(bloods_drop_image, (int(blood_width ), int(blood_height )))

    # Position the blood drop below the lip, but use proportions based on the image size
    drip_x = int(right_nostril[0] - blood_drop_resized.shape[1] // 2)
    drip_y = int(right_nostril[1] - blood_drop_resized.shape[1] * 0.1)  # Position 5% below the lip

    # Ensure drip_x and drip_y are within the bounds of the image dimensions
    drip_x = max(0, min(drip_x, image.shape[1] - blood_drop_resized.shape[1]))
    drip_y = max(0, min(drip_y, image.shape[0] - blood_drop_resized.shape[0]))

    # Get the region of interest (ROI) on the face where the blood drop will be placed
    roi_height, roi_width = blood_drop_resized.shape[:2]
    roi = image[drip_y:drip_y + roi_height, drip_x:drip_x + roi_width]

    # If blood drop image has transparency (4 channels), blend it with the face image
    if blood_drop_resized.shape[2] == 4:  # If there's an alpha channel (transparency)
        # Split the channels of the blood drop image (BGR and Alpha)
        blood_bgr = blood_drop_resized[:, :, :3]
        alpha_channel = blood_drop_resized[:, :, 3] / 255.0

        # Invert alpha channel for blending
        inv_alpha_channel = 1.0 - alpha_channel

        # Blend the blood drop with the ROI
        for c in range(0, 3):  # For each color channel
            roi[:, :, c] = (alpha_channel * blood_bgr[:, :, c] +
                            inv_alpha_channel * roi[:, :, c])
    else:
        # If no transparency, directly overlay the image (BGR format)
        image[drip_y:drip_y + roi_height, drip_x:drip_x + roi_width] = blood_drop_resized

    return image

def add_wound_mark_below_eyes(image, face, landmarks, wound_image):
    # Load the wound image as a NumPy array if it hasn't been loaded yet
    # wound_image = cv2.imread('path_to_wound_image.png', cv2.IMREAD_UNCHANGED)
    height, width = image.shape[:2]

    # Get eye landmarks
    left_eye_left = landmarks.part(36)  # Left eye left corner
    left_eye_right = landmarks.part(39)  # Left eye right corner
    right_eye_left = landmarks.part(42)  # Right eye left corner
    right_eye_right = landmarks.part(45)  # Right eye right corner

    # Resize wound image if necessary
    face_width = face.right() - face.left()  # Estimate face width
    wound_width = int(face_width * 0.4)  # 20% of face width for scar
    wound_height = int(wound_width * 0.7)  # Adjust the height of the scar


    left_eye_center = (
        (left_eye_left.x + left_eye_right.x) // 2,
        (left_eye_left.y + left_eye_right.y) // 2 + int(0.09 * face_width)  # Position below the eye
    )
    right_eye_center = (
        (right_eye_left.x + right_eye_right.x) // 2,
        (right_eye_left.y + right_eye_right.y) // 2 + int(0.09 * face_width )  # Position below the eye
    )
    # Resize the scar image to fit the forehead
    # Resize blood drop image if necessary

    wound_resized = cv2.resize(wound_image, (int(wound_width ), int(wound_height )))

    # Get the height and width of the resized wound image
    wound_roi_height, wound_roi_width = wound_resized.shape[:2]

    # Calculate positions to overlay the wound marks
    left_x = left_eye_center[0] - wound_roi_width // 2
    left_y = left_eye_center[1] - wound_roi_height // 2

    right_x = right_eye_center[0] - wound_roi_width // 2
    right_y = right_eye_center[1] - wound_roi_height // 2

    # Ensure the coordinates are within the image boundaries
    left_x = max(0, min(left_x, width - wound_roi_width))
    left_y = max(0, min(left_y, height - wound_roi_height))

    right_x = max(0, min(right_x, width - wound_roi_width))
    right_y = max(0, min(right_y, height - wound_roi_height))

    # Overlay the wound image on the left eye
    roi_left = image[left_y:left_y + wound_roi_height, left_x:left_x + wound_roi_width]
    if wound_resized.shape[2] == 4:  # If there's an alpha channel (transparency)
        wound_bgr = wound_resized[:, :, :3]
        alpha_channel = wound_resized[:, :, 3] / 255.0
        inv_alpha_channel = 1.0 - alpha_channel
        for c in range(3):  # For each color channel
            roi_left[:, :, c] = (alpha_channel * wound_bgr[:, :, c] + inv_alpha_channel * roi_left[:, :, c])
    else:
        image[left_y:left_y + wound_roi_height, left_x:left_x + wound_roi_width] = wound_resized

    # Overlay the wound image on the right eye
    roi_right = image[right_y:right_y + wound_roi_height, right_x:right_x + wound_roi_width]
    if wound_resized.shape[2] == 4:  # If there's an alpha channel (transparency)
        wound_bgr = wound_resized[:, :, :3]
        alpha_channel = wound_resized[:, :, 3] / 255.0
        inv_alpha_channel = 1.0 - alpha_channel
        for c in range(3):  # For each color channel
            roi_right[:, :, c] = (alpha_channel * wound_bgr[:, :, c] + inv_alpha_channel * roi_right[:, :, c])
    else:
        image[right_y:right_y + wound_roi_height, right_x:right_x + wound_roi_width] = wound_resized

    return image


def apply_swollen_eye(image, face, landmarks, swollens_eye_image):
    # Get dimensions of the input image
    height, width = image.shape[:2]

    # Get eye landmarks for the right eye
    right_eye_left = landmarks.part(42)  # Right eye left corner
    right_eye_right = landmarks.part(45)  # Right eye right corner
    right_eye_top = landmarks.part(43)    # Right eye top center

    # Calculate the position for the swollen effect
    swollen_position = (
        (right_eye_left.x + right_eye_right.x) // 2,
        right_eye_top.y + int(0.009 * height)   # Position just above the eye
    )

    # Resize wound image if necessary
    face_width = face.right() - face.left()  # Estimate face width
    swollen_width = int(face_width * 0.3)  # 20% of face width for scar
    swollen_height = int(swollen_width * 0.55)  # Adjust the height of the scar


    swollen_resized = cv2.resize(swollens_eye_image, (int(swollen_width ), int(swollen_height )))

    # Get the dimensions of the resized swollen image
    swollen_roi_height, swollen_roi_width = swollen_resized.shape[:2]

    # Calculate the position to overlay the swollen eye
    swollen_x = swollen_position[0] - swollen_roi_width // 2
    swollen_y = swollen_position[1] - swollen_roi_height // 2

    # Ensure the coordinates are within the image boundaries
    swollen_x = max(0, min(swollen_x, width - swollen_roi_width))
    swollen_y = max(0, min(swollen_y, height - swollen_roi_height))

    # Overlay the swollen eye image
    roi = image[swollen_y:swollen_y + swollen_roi_height, swollen_x:swollen_x + swollen_roi_width]
    if swollen_resized.shape[2] == 4:  # If there's an alpha channel (transparency)
        swollen_bgr = swollen_resized[:, :, :3]
        alpha_channel = swollen_resized[:, :, 3] / 255.0
        inv_alpha_channel = 1.0 - alpha_channel
        for c in range(3):  # For each color channel
            roi[:, :, c] = (alpha_channel * swollen_bgr[:, :, c] + inv_alpha_channel * roi[:, :, c])
    else:
        image[swollen_y:swollen_y + swollen_roi_height, swollen_x:swollen_x + swollen_roi_width] = swollen_resized

    return image


def apply_injuries(image, face, landmarks, bloods_drop_image):
    # Apply red eyes
    image = apply_red_eye(image, landmarks)


    # Add blood drip from the lips
    image = add_blood_drop_image(image, face, landmarks, bloods_drop_image)

    return image


def apply_bullet_scar(image, face, landmarks, bullet_scar_image):
    # Get dimensions of the input image
    height, width = image.shape[:2]

    left_lip_bottom = landmarks.part(48)  # Landmark index for the bottom left lip

    # Calculate the position for the bullet scar

    # Resize wound image if necessary
    face_width = face.right() - face.left()  # Estimate face width
    scar_width = int(face_width * 0.2)  # 20% of face width for scar
    scar_height = int(scar_width * 0.6)  # Adjust the height of the scar
    scar_position = (left_lip_bottom.x, left_lip_bottom.y - int(0.06 * face_width ))  # Position just below the lip


    scar_resized = cv2.resize(bullet_scar_image, (int(scar_width ), int(scar_height )))

    scar_roi_height, scar_roi_width = scar_resized.shape[:2]

    # Calculate the position to overlay the bullet scar
    scar_x = scar_position[0] - scar_roi_width // 2
    scar_y = scar_position[1] - scar_roi_height // 2

    # Ensure the coordinates are within the image boundaries
    scar_x = max(0, min(scar_x, width - scar_roi_width))
    scar_y = max(0, min(scar_y, height - scar_roi_height))

    # Overlay the bullet scar image
    roi = image[scar_y:scar_y + scar_roi_height, scar_x:scar_x + scar_roi_width]
    if scar_resized.shape[2] == 4:  # If there's an alpha channel (transparency)
        scar_bgr = scar_resized[:, :, :3]
        alpha_channel = scar_resized[:, :, 3] / 255.0
        inv_alpha_channel = 1.0 - alpha_channel
        for c in range(3):  # For each color channel
            roi[:, :, c] = (alpha_channel * scar_bgr[:, :, c] + inv_alpha_channel * roi[:, :, c])
    else:
        image[scar_y:scar_y + scar_roi_height, scar_x:scar_x + scar_roi_width] = scar_resized

    return image

def overlay_image(face_image, image, x, y):
    """Place the scar image at position (x, y) on the face image with alpha blending"""
    scar_h, scar_w = image.shape[:2]

    # Extract the region of interest (ROI) from the face where the scar will be placed
    roi = face_image[y:y+scar_h, x:x+scar_w]

    # Split the scar image into color and alpha channels
    scar_bgr = image[:, :, :3]
    scar_alpha = image[:, :, 3] / 255.0

    # Blend the scar image with the ROI using the alpha channel
    for c in range(0, 3):  # Iterate over color channels
        roi[:, :, c] = (scar_alpha * scar_bgr[:, :, c] + (1.0 - scar_alpha) * roi[:, :, c])

    # Put the blended region back into the face image
    face_image[y:y+scar_h, x:x+scar_w] = roi

def apply_punch_mark(image, landmarks, punch_mark_image):
    # Calculate face dimensions using facial landmarks
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between the leftmost and rightmost points
    face_height = landmarks.part(8).y - landmarks.part(19).y  # Rough estimate for face height

    # Define the size of the punch mark as a proportion of face dimensions
    punch_width = int(face_width * 0.2)  # 20% of face width
    punch_height = int(face_height * 0.15)  # 15% of face height

    # Validate the dimensions
    if punch_width <= 0 or punch_height <= 0:
        print("Error: Invalid punch mark dimensions.")
        return image

    # Resize the punch mark image to fit on the forehead
    resized_punch = cv2.resize(punch_mark_image, (punch_width, punch_height))

    # Define position for the left side of the forehead using landmarks
    # Landmark points around 18, 19 define the left side of the forehead
    forehead_x = landmarks.part(19).x  # Near the left side of the forehead
    forehead_y = landmarks.part(18).y - int(face_height * 0.2)  # A bit higher up for the forehead

    # Subtract half the punch width/height to center it on the forehead
    punch_x = forehead_x - punch_width // 2
    punch_y = forehead_y - punch_height // 2

    # Ensure the coordinates are within the image boundaries
    if punch_x + punch_width > image.shape[1]:
        punch_x = image.shape[1] - punch_width
    if punch_y + punch_height > image.shape[0]:
        punch_y = image.shape[0] - punch_height
    if punch_x < 0:
        punch_x = 0
    if punch_y < 0:
        punch_y = 0

    # Add the punch mark image onto the forehead (simple paste for now; alpha blending can be added)
    overlay_image(image, resized_punch, punch_x, punch_y)

    return image


def apply_slap_mark(image, landmarks,slap_mark_image, side='left'):
    # Calculate face dimensions using facial landmarks
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between the leftmost and rightmost points
    face_height = landmarks.part(8).y - landmarks.part(19).y  # Rough estimate for face height

    # Define the size of the slap mark as a proportion of face dimensions
    slap_width = int(face_width * 0.3)  # 40% of face width to cover a large area of the cheek
    slap_height = int(face_height * 0.4)  # 25% of face height for vertical coverage

    # Validate the dimensions
    if slap_width <= 0 or slap_height <= 0:
        print("Error: Invalid slap mark dimensions.")
        return image

    # Resize the slap mark image to fit the face size
    resized_slap_mark = cv2.resize(slap_mark_image, (slap_width, slap_height))

    # Define position for the slap mark based on the chosen side (left or right)
    if side == 'left':
        cheek_landmark = landmarks.part(3)  # Use part(2) for the left cheek

        # Position the slap mark on the cheek
        slap_x = int(cheek_landmark.x - (face_width * 0.1)) + slap_width // 2  # Center it on the cheek horizontally
        slap_y = int(cheek_landmark.y - (face_height * 0.1)) - slap_height // 2  # Center it vertically on the cheek
    else:
        cheek_landmark = landmarks.part(14)  # Use part(14) for the right cheek

        # Position the slap mark on the cheek
        slap_x = int(cheek_landmark.x - (face_width * 0.5)) + slap_width // 2  # Center it on the cheek horizontally
        slap_y = cheek_landmark.y - slap_height // 2  # Center it vertically on the cheek


    # Ensure the coordinates are within the image boundaries
    if slap_x + slap_width > image.shape[1]:
        slap_x = image.shape[1] - slap_width
    if slap_y + slap_height > image.shape[0]:
        slap_y = image.shape[0] - slap_height
    if slap_x < 0:
        slap_x = 0
    if slap_y < 0:
        slap_y = 0

    # Add the slap mark image onto the face (simple paste for now; alpha blending can be added)
    overlay_image(image, resized_slap_mark, slap_x, slap_y)

    return image

#blister right lips cheek mark
def apply_blister_mark_right_cheek(image, landmarks, blister_mark_image):
    # Calculate face dimensions using facial landmarks
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between the leftmost and rightmost points
    face_height = landmarks.part(8).y - landmarks.part(19).y  # Rough estimate for face height

    # Define the size of the blister as a proportion of face dimensions
    blister_width = int(face_width * 0.15)  # 15% of face width
    blister_height = int(face_height * 0.15)  # 15% of face height

    # Validate the dimensions
    if blister_width <= 0 or blister_height <= 0:
        print("Error: Invalid blister dimensions.")
        return image

    # Resize the blister image to fit the face size
    resized_blister = cv2.resize(blister_mark_image, (blister_width, blister_height))

    # Define position for the left side of the bottom lip
    # Using landmark points around the mouth
    left_lip_landmark = landmarks.part(48)  # Left corner of the mouth
    bottom_lip_landmark = landmarks.part(57)  # Bottom of the lower lip

    # Position the blister on the left side of the bottom lip
    blister_x = left_lip_landmark.x - blister_width // 2  # Shift it slightly left
    blister_y = bottom_lip_landmark.y - (blister_height // 2)  # Slightly above the bottom lip

    # Ensure the coordinates are within the image boundaries
    if blister_x + blister_width > image.shape[1]:
        blister_x = image.shape[1] - blister_width
    if blister_y + blister_height > image.shape[0]:
        blister_y = image.shape[0] - blister_height
    if blister_x < 0:
        blister_x = 0
    if blister_y < 0:
        blister_y = 0

    # Add the blister image onto the face (simple paste for now; alpha blending can be added)
    overlay_image(image, resized_blister, blister_x, blister_y)

    return image

def apply_bullet_eyes(image, landmarks,blackmark_mark_image):
    # Calculate face dimensions using facial landmarks
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between the leftmost and rightmost points
    face_height = landmarks.part(8).y - landmarks.part(19).y  # Rough estimate for face height

    # Define the size of the bullet scar as a proportion of face dimensions
    scar_width = int(face_width * 0.5)  # 15% of face width
    scar_height = int(face_height * 0.5)  # 15% of face height

    # Validate the dimensions
    if scar_width <= 0 or scar_height <= 0:
        print("Error: Invalid scar dimensions.")
        return image

    # Resize the bullet scar image to fit the eye area
    resized_bullet_scar = cv2.resize(blackmark_mark_image, (scar_width, scar_height))

    # Define position for the eye area using eye landmarks
    # Use landmarks for the left or right eye; example for the right eye
    right_eye_x = landmarks.part(36).x  # Right eye's left corner
    right_eye_y = landmarks.part(36).y  # Right eye's top corner

    # Subtract half the scar width/height to center it on the eye
    scar_x = right_eye_x - scar_width // 2
    scar_y = right_eye_y - scar_height // 2

    # Ensure the coordinates are within the image boundaries
    if scar_x + scar_width > image.shape[1]:
        scar_x = image.shape[1] - scar_width
    if scar_y + scar_height > image.shape[0]:
        scar_y = image.shape[0] - scar_height
    if scar_x < 0:
        scar_x = 0
    if scar_y < 0:
        scar_y = 0

    # Add the bullet scar image onto the eye area (simple paste; alpha blending can be added)
    overlay_image(image, resized_bullet_scar, scar_x, scar_y)

    return image


def apply_bite_mark(image, landmarks, bite_mark_image, position='cheek'):
    # Calculate face dimensions using facial landmarks
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between the leftmost and rightmost points
    face_height = landmarks.part(8).y - landmarks.part(19).y  # Rough estimate for face height

    # Define the size of the bite mark as a proportion of face dimensions
    bite_width = int(face_width * 0.2)  # 15% of face width
    bite_height = int(face_height * 0.2)  # 10% of face height

    # Validate the dimensions
    if bite_width <= 0 or bite_height <= 0:
        print("Error: Invalid bite mark dimensions.")
        return image

    # Resize the bite mark image to fit the face size
    resized_bite_mark = cv2.resize(bite_mark_image, (bite_width, bite_height))

    # Define position for the bite mark based on the chosen area (cheek, jaw, etc.)
    if position == 'cheek':
        # Use part(13) for the right cheek, part(3) for left cheek
        cheek_landmark = landmarks.part(13)  # Change to part(3) for left cheek if needed
    elif position == 'jaw':
        # Use part(7) for chin/jaw area
        cheek_landmark = landmarks.part(7)
    elif position == 'forehead':
        # Use part(19) for the forehead
        cheek_landmark = landmarks.part(19)
    else:
        # Default to part(13) for the right cheek
        cheek_landmark = landmarks.part(13)

    if position == 'cheek':

        # Subtract half the bite mark width/height to center it at the desired landmark
        bite_x = int(cheek_landmark.x - (face_width * 0.9)) + bite_width // 2
        bite_y = int(cheek_landmark.y - (face_height * 0.1)) - bite_height // 2

    else:

        # Subtract half the bite mark width/height to center it at the desired landmark
        bite_x = int(cheek_landmark.x - (face_width * 0.1)) + bite_width // 2
        bite_y = int(cheek_landmark.y - (face_height * 0.1)) - bite_height // 2

    # Ensure the coordinates are within the image boundaries
    if bite_x + bite_width > image.shape[1]:
        bite_x = image.shape[1] - bite_width
    if bite_y + bite_height > image.shape[0]:
        bite_y = image.shape[0] - bite_height
    if bite_x < 0:
        bite_x = 0
    if bite_y < 0:
        bite_y = 0

    # Add the bite mark image onto the face (simple paste for now; alpha blending can be added)
    overlay_image(image, resized_bite_mark, bite_x, bite_y)

    return image

def apply_knife_stab(image,landmarks, stab_mark_image):
    # Calculate face dimensions using facial landmarks
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between the leftmost and rightmost points
    face_height = landmarks.part(8).y - landmarks.part(19).y  # Rough estimate for face height

    # Define the size of the knife as a proportion of face dimensions
    knife_width = int(face_width * 0.8)  # 40% of face width
    knife_height = int(face_height * 0.9)  # 50% of face height

    # Validate the dimensions
    if knife_width <= 0 or knife_height <= 0:
        print("Error: Invalid knife dimensions.")
        return image

    # Resize the knife image to fit the head area
    resized_knife = cv2.resize(stab_mark_image, (knife_width, knife_height))

    # Define position for the left side of the head
    # We'll use landmark point 1 for the left temple or near the left side of the head
    left_head_x = landmarks.part(1).x  # Use point 1 for left side
    left_head_y = landmarks.part(19).y  # Near the top of the forehead

    # Adjust position for left of the head by subtracting more of the width
    knife_x = left_head_x - knife_width // 2 # Move to the left of the head
    knife_y = left_head_y - knife_height // 2  # Center vertically around the forehead

    # Ensure the coordinates are within the image boundaries
    if knife_x + knife_width > image.shape[1]:
        knife_x = image.shape[1] - knife_width
    if knife_y + knife_height > image.shape[0]:
        knife_y = image.shape[0] - knife_height
    if knife_x < 0:
        knife_x = 0
    if knife_y < 0:
        knife_y = 0

    # Add the knife image onto the left side of the head (simple paste; alpha blending can be added)
    overlay_image(image, resized_knife, knife_x, knife_y)

    return image


def apply_patch_random(image, landmarks, patch_mark_image):

    # Calculate face dimensions using facial landmarks
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between the leftmost and rightmost points
    face_height = landmarks.part(8).y - landmarks.part(19).y  # Rough estimate for face height

    # Define the size of the bandage as a proportion of face dimensions
    bandage_width = int(face_width * 0.25)  # 15% of face width
    bandage_height = int(face_height * 0.25)  # 10% of face height

    # Validate the dimensions
    if bandage_width <= 0 or bandage_height <= 0:
        print("Error: Invalid bandage dimensions.")
        return image

    # Resize the bandage image to fit the face size
    resized_bandage = cv2.resize(patch_mark_image, (bandage_width, bandage_height))

    # Randomly select a landmark point for the bandage placement
    random_landmark_index = random.choice([ 51])  # Select key points on the face
    random_landmark = landmarks.part(random_landmark_index)

    # Define the position based on the selected landmark point
    bandage_x = random_landmark.x - bandage_width // 2
    bandage_y = random_landmark.y - bandage_height // 2

    # Ensure the coordinates are within the image boundaries
    if bandage_x + bandage_width > image.shape[1]:
        bandage_x = image.shape[1] - bandage_width
    if bandage_y + bandage_height > image.shape[0]:
        bandage_y = image.shape[0] - bandage_height
    if bandage_x < 0:
        bandage_x = 0
    if bandage_y < 0:
        bandage_y = 0

    # Add the bandage image onto the face at the random location (simple paste for now; alpha blending can be added)
    overlay_image(image, resized_bandage, bandage_x, bandage_y)

    return image


def apply_black_mark(image, landmarks, blacks_mark_image, position='cheek'):
    # Calculate face dimensions using facial landmarks
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between the leftmost and rightmost points
    face_height = landmarks.part(8).y - landmarks.part(19).y  # Rough estimate for face height

    # Define the size of the bite mark as a proportion of face dimensions
    bite_width = int(face_width * 0.2)  # 15% of face width
    bite_height = int(face_height * 0.2)  # 10% of face height

    # Validate the dimensions
    if bite_width <= 0 or bite_height <= 0:
        print("Error: Invalid bite mark dimensions.")
        return image

    # Resize the bite mark image to fit the face size
    resized_bite_mark = cv2.resize(blacks_mark_image, (bite_width, bite_height))

    # Define position for the bite mark based on the chosen area (cheek, jaw, etc.)
    if position == 'cheek':
        # Use part(13) for the right cheek, part(3) for left cheek
        cheek_landmark = landmarks.part(13)  # Change to part(3) for left cheek if needed
    elif position == 'jaw':
        # Use part(7) for chin/jaw area
        cheek_landmark = landmarks.part(7)
    elif position == 'forehead':
        # Use part(19) for the forehead
        cheek_landmark = landmarks.part(19)
    else:
        # Default to part(13) for the right cheek
        cheek_landmark = landmarks.part(13)

    if position == 'cheek':

        # Subtract half the bite mark width/height to center it at the desired landmark
        bite_x = int(cheek_landmark.x - (face_width * 0.9)) + bite_width // 2
        bite_y = int(cheek_landmark.y - (face_height * 0.1)) - bite_height // 2

    else:

        # Subtract half the bite mark width/height to center it at the desired landmark
        bite_x = int(cheek_landmark.x - (face_width * 0.1)) + bite_width // 2
        bite_y = int(cheek_landmark.y - (face_height * 0.1)) - bite_height // 2

    # Ensure the coordinates are within the image boundaries
    if bite_x + bite_width > image.shape[1]:
        bite_x = image.shape[1] - bite_width
    if bite_y + bite_height > image.shape[0]:
        bite_y = image.shape[0] - bite_height
    if bite_x < 0:
        bite_x = 0
    if bite_y < 0:
        bite_y = 0

    # Add the bite mark image onto the face (simple paste for now; alpha blending can be added)
    overlay_image(image, resized_bite_mark, bite_x, bite_y)

    return image

def apply_big_eye(image, landmarks,big_eye_mark_image):
    # Calculate face dimensions using facial landmarks
    face_width = landmarks.part(16).x - landmarks.part(0).x  # Distance between the leftmost and rightmost points
    face_height = landmarks.part(8).y - landmarks.part(19).y  # Rough estimate for face height

    # Define the size of the injury as a proportion of face dimensions
    injury_width = int(face_width * 0.5)  # 30% of face width
    injury_height = int(face_height * 0.5)  # 20% of face height

    # Validate the dimensions
    if injury_width <= 0 or injury_height <= 0:
        print("Error: Invalid injury dimensions.")
        return image

    # Resize the injury image to fit over the eye
    resized_injury = cv2.resize(big_eye_mark_image, (injury_width, injury_height))

    # Define the position for the right eye using landmarks
    # Landmark points 36 to 41 usually define the right eye area
    right_eye_x = landmarks.part(36).x  # The left corner of the right eye
    right_eye_y = landmarks.part(37).y  # The top point of the right eye

    # Subtract half the injury width/height to center it over the eye
    injury_x = right_eye_x - injury_width // 2
    injury_y = right_eye_y - injury_height // 2

    # Ensure the coordinates are within the image boundaries
    if injury_x + injury_width > image.shape[1]:
        injury_x = image.shape[1] - injury_width
    if injury_y + injury_height > image.shape[0]:
        injury_y = image.shape[0] - injury_height
    if injury_x < 0:
        injury_x = 0
    if injury_y < 0:
        injury_y = 0

    # Add the injury image onto the face at the right eye position (simple paste for now; alpha blending can be added)
    overlay_image(image, resized_injury, injury_x, injury_y)

    return image

def resize_image(image, scale_factor=2.0):
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    return cv2.resize(image, (width, height))

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def detect_face_with_mtcnn(image):
    detector = MTCNN()

    # Detect faces in the image
    faces = detector.detect_faces(image)

    # If no faces are detected, return the original image
    if len(faces) == 0:
        print("No faces detected")
        return image

    # Loop through the faces and mark the landmarks
    for face in faces:
        # Get face bounding box
        x, y, width, height = face['box']
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)

        # Get facial landmarks
        landmarks = face['keypoints']

        # Draw circles on the landmarks
        for key, point in landmarks.items():
            cv2.circle(image, point, 2, (0, 255, 0), -1)

    return image
# Function to detect and mark facial features
def detect_facial_features(image_path, output_dir, scar_density):
    # Load the image

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)


    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)

        # Draw circles for facial landmarks
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Green circles for the landmarks

        # Apply resizing to the image before detection

        image_with_landmarks = detect_face_with_mtcnn(image)
        cv2.imshow('Face with landmarks', image_with_landmarks)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Draw cheeks
        left_cheek_x = landmarks.part(1).x
        left_cheek_y = landmarks.part(1).y

        right_cheek_x = landmarks.part(15).x
        right_cheek_y = landmarks.part(15).y

        # Draw top of the cheeks (averaging below the eye landmarks)
        left_top_cheek_x = landmarks.part(1).x
        left_top_cheek_y = (landmarks.part(42).y + landmarks.part(39).y) // 2  # Between left eye and left cheek

        right_top_cheek_x = landmarks.part(15).x
        right_top_cheek_y = (landmarks.part(45).y + landmarks.part(36).y) // 2  # Between right eye and right cheek


        # Draw middle of the forehead
        forehead_x = (landmarks.part(19).x + landmarks.part(24).x) // 2
        forehead_y = landmarks.part(19).y - 0  # Adjust as needed
        # Increase the offset to move up


        # Cheeks: Estimating using area below eyes and beside the nose
        # Draw chin
        chin_x = landmarks.part(8).x
        chin_y = landmarks.part(8).y
        blood_drop_position = (left_cheek_x , left_cheek_y )  # Adjust position for left cheek
        print(scar_density)
        if scar_density == 1:
            image = apply_blood_left_cheek_drop(image, landmarks, blood_drop_image)  # Scale can be adjusted
            image = apply_burn_mark(image, landmarks, burn_image)
            image = apply_injuries(image, face, landmarks, bloods_drop_image)
            image = apply_swollen_eye(image, face, landmarks, swollens_eye_image)
            image = apply_blood_drip_right_cheek(image,landmarks, blood_drip_image)
            image = apply_punch_mark(image, landmarks, punch_mark_image)
            image = apply_slap_mark(image, landmarks, slap_mark_image, side='left')
            image = apply_knife_stab(image, landmarks, stab_mark_image)
            image = apply_black_mark(image,landmarks, blacks_mark_image, position='jaw')
            image = apply_big_cut_between_eyebrow(image,landmarks,big_cut_mark_image)

        elif scar_density == 2:
            image = apply_cut_left_cheek(image, landmarks, cut_image)
            image = apply_burn_mark(image, landmarks, burn_image)
            image = apply_injuries(image, face, landmarks, bloods_drop_image)
            image = apply_swollen_eye(image, face, landmarks, swollens_eye_image)
            image = add_wound_mark_below_eyes(image, face, landmarks, wound_image)
            image = apply_punch_mark(image, landmarks, punch_mark_image)
            image = apply_bite_mark(image, landmarks, bite_mark_image, position='cheek')
            image = apply_bullet_eyes(image, landmarks, blackmark_mark_image)
            image = apply_bullet_right_cheek(image, landmarks, scar_image)
            image = apply_patch_random(image, landmarks, patch_mark_image)
            image = apply_black_mark(image,landmarks, blacks_mark_image, position='cheek')

        elif scar_density == 3:
            image = apply_injuries(image, face, landmarks, bloods_drop_image)
            image = apply_swollen_eye(image, face, landmarks, swollens_eye_image)
            image = apply_stitch_forehead(image, face, landmarks, swollen_eye_image)
            image = apply_patch_random(image, landmarks, patch_mark_image)
            image = apply_wound_right_forehead(image, landmarks, stitch_image)
            image = apply_slap_mark(image, landmarks, slap_mark_image, side='right')
            image = apply_knife_stab(image, landmarks, stab_mark_image)

        elif scar_density == 4:
            image = apply_cut_between_eyebrow(image, landmarks, cut_image)
            image = apply_burn_mark(image, landmarks, burn_image)
            image = apply_injuries(image, face, landmarks, bloods_drop_image)
            image = apply_swollen_eye(image, face, landmarks, swollens_eye_image)
            image = apply_blood_left_cheek_drop(image, landmarks, blood_drop_image)
            image = apply_bullet_right_cheek(image, landmarks, scar_image)
            image = apply_bullet_eyes(image, landmarks, blackmark_mark_image)
            image = apply_punch_mark(image, landmarks, punch_mark_image)
            image = apply_wound_right_forehead(image, landmarks, stitch_image)
            image = apply_bite_mark(image, landmarks, bite_mark_image, position='jaw')
            image = apply_big_eye(image, landmarks, big_eye_mark_image)
        elif scar_density == 5:
            image = apply_burn_mark(image, landmarks, burn_image)
            image = apply_injuries(image, face, landmarks, bloods_drop_image)
            image = apply_swollen_eye(image, face, landmarks, swollens_eye_image)
            image = apply_bullet_scar(image, face, landmarks, bullet_scar_image)
            image = apply_stitch_forehead(image, face, landmarks, swollen_eye_image)
            image = apply_slap_mark(image, landmarks, slap_mark_image, side='right')
            image = apply_patch_random(image, landmarks, patch_mark_image)
            image = apply_knife_stab(image, landmarks, stab_mark_image)
            image = apply_big_cut_between_eyebrow(image,landmarks,big_cut_mark_image)

        elif scar_density == 6:
            image = apply_injuries(image, face, landmarks, bloods_drop_image)
            image = apply_burn_mark(image, landmarks, burn_image)
            image = apply_mark_right_cheek(image, landmarks, mark_image)
            image = apply_injuries(image, face, landmarks, bloods_drop_image)
            image = apply_swollen_eye(image, face, landmarks, swollens_eye_image)
            image = apply_blister_mark_right_cheek(image, landmarks, blister_mark_image)
            image = apply_wound_right_forehead(image, landmarks, stitch_image)
            image = apply_punch_mark(image, landmarks, punch_mark_image)
            image = apply_bullet_eyes(image, landmarks, blackmark_mark_image)


        elif scar_density == 7:
            image = add_wound_mark_below_eyes(image, face, landmarks, wound_image)
            image = apply_burn_mark(image, landmarks, burn_image)
            image = apply_injuries(image, face, landmarks, bloods_drop_image)
            image = apply_swollen_eye(image, face, landmarks, swollens_eye_image)
            image = apply_stitch_forehead(image, face, landmarks, swollen_eye_image)
            image = apply_blood_left_cheek_drop(image, landmarks, blood_drop_image)
            image = apply_blister_mark_right_cheek(image, landmarks, blister_mark_image)
            image = apply_knife_stab(image, landmarks, stab_mark_image)
            image = apply_patch_random(image, landmarks, patch_mark_image)
            image = apply_big_cut_between_eyebrow(image,landmarks,big_cut_mark_image)
            image = apply_big_eye(image,landmarks,big_eye_mark_image)

        elif scar_density == 8:
            image = apply_bullet_right_cheek(image, landmarks, scar_image)
            image = apply_burn_mark(image, landmarks, burn_image)
            image = apply_injuries(image, face, landmarks, bloods_drop_image)
            image = apply_swollen_eye(image, face, landmarks, swollens_eye_image)
            image = apply_cut_between_eyebrow(image, landmarks, cut_image)
            image = apply_blister_mark_right_cheek(image,landmarks, blister_mark_image)
            image = apply_wound_right_forehead(image, landmarks, stitch_image)
            image = apply_bullet_eyes(image, landmarks, blackmark_mark_image)
            image = apply_big_eye(image, landmarks, big_eye_mark_image)
        elif scar_density == 0:

            #cheeks
            image = apply_blood_left_cheek_drop(image, landmarks, blood_drop_image)
            image = apply_black_mark(image,landmarks, blacks_mark_image, position='cheek')

            image = apply_bullet_right_cheek(image, landmarks, scar_image)
            image = apply_bullet_eyes(image, landmarks, blackmark_mark_image)
            #lips
            image = apply_blister_mark_right_cheek(image,landmarks, blister_mark_image)
            image = apply_knife_stab(image, landmarks, stab_mark_image)
            #forehead
            image = apply_stitch_forehead(image, face, landmarks, swollen_eye_image)
            image = apply_cut_between_eyebrow(image, landmarks, cut_image)
            image = apply_bullet_right_cheek(image, landmarks, scar_image)
            image = apply_swollen_eye(image, face, landmarks, swollens_eye_image)


    # Prepare to save the output image
    print(os.path.basename(image_path))
    output_filename = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_filename, image)
    print(f"Processed image saved as: {output_filename}")

# Main function to process a directory of images
def process_directory(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all images in the input directory
    scar_density = 0

    for filename in os.listdir(input_dir):

        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Supported image formats
            image_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")


            print((scar_density) % 9)
            detect_facial_features(image_path, output_dir,scar_density % 9 )
            scar_density += 1


# Example usage
input_directory = "C:/Users/DoICT/PycharmProjects/addfaceinjury/input_images/"  # Update this path
output_directory = "C:/Users/DoICT/PycharmProjects/addfaceinjury/output_images/"  # Update this path
process_directory(input_directory, output_directory)
