import dlib
import cv2
import os

# Get the path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
predictor_path = os.path.join(current_dir, "shape_predictor_68_face_landmarks.dat")  # Update this line

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(predictor_path)

def get_facial_landmarks(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_image)

    if len(faces) > 0:
        face = faces[0]
        landmarks = landmark_predictor(gray_image, face)
        landmarks_points = [(p.x, p.y) for p in landmarks.parts()]
        return landmarks_points
    else:
        raise ValueError("No face detected in the image.")