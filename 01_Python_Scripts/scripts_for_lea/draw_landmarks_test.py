from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green    sdvccx 

print(type(HANDEDNESS_TEXT_COLOR))

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx] 

        print(handedness)
        print(handedness[0].category_name)

        print(range(len(hand_landmarks)))
        print(type(hand_landmarks))


        print('done')

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        #print(hand_landmarks_proto)

        hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
        solutions.drawing_utils.draw_landmarks(annotated_image,hand_landmarks_proto,solutions.hands.HAND_CONNECTIONS,solutions.drawing_styles.get_default_hand_landmarks_style(),solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape

        x_coordinates = [landmark.x for landmark in hand_landmarks]
        print(x_coordinates)
        print(range(len(x_coordinates)))
        print(type(x_coordinates))
        y_coordinates = [landmark.y for landmark in hand_landmarks]

        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw rectangle around hand
        min_y = int(min(y_coordinates) * height)
        box_x = int(max(x_coordinates) * width)
        box_y = int(max(y_coordinates) * height)
        padding = 20

        cv2.rectangle(annotated_image, (text_x - padding, min_y - padding), (box_x + padding, box_y + padding), (255,0,0), 2)

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        
    return annotated_image


# STEP 1: Import the necessary modules.


# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# cap = cv2.VideoCapture(0)

# while True:
#     success, img = cap.read()
#     cv2.imshow('live', img)
#     cv2.waitKey(1)

# STEP 3: Load the input image.
#image2 = mp.Image.create_from_file('./data/save97.jpg')
image2 = mp.Image.create_from_file('./no_hands.jpg')

#image2 = cv2.imread('./data/save97.jpg')
print('this is imag')
print(image2)

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image2)
print('here starts detection result')
print(detection_result)
print('here ends detection result')

# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image2.numpy_view(), detection_result)

# print(annotated_image.shape)

img_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

cv2.imshow('image', img_rgb)
k = cv2.waitKey(0)
