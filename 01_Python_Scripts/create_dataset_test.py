# This script shows the hand landmarks when creating the dataset.



import cv2
import os

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

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx] 

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        #print(hand_landmarks_proto)

        hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
        solutions.drawing_utils.draw_landmarks(annotated_image,hand_landmarks_proto,solutions.hands.HAND_CONNECTIONS,solutions.drawing_styles.get_default_hand_landmarks_style(),solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape

        x_coordinates = [landmark.x for landmark in hand_landmarks]
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
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
#image2 = mp.Image.create_from_file('./no_hands.jpg')

# # STEP 4: Detect hand landmarks from the input image.
# detection_result = detector.detect(image2)

# # STEP 5: Process the classification result. In this case, visualize it.
# annotated_image = draw_landmarks_on_image(image2.numpy_view(), detection_result)

# # print(annotated_image.shape)

# img_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# cv2.imshow('image', img_rgb)
# k = cv2.waitKey(0)

# ------------------ Functions to create dataset ------------------


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 100

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    while True:
        success, img = cap.read()

        image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data= img)

        detection_result = detector.detect(image2)
        annotated_image = draw_landmarks_on_image(image2.numpy_view(), detection_result)

        cv2.putText(annotated_image, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        
        cv2.imshow('live', annotated_image)
        #cv2.waitKey(500)

        #if cv2.waitKey(20) & 0xFF==ord('d'):
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0

    while(counter < dataset_size):
        ret, img = cap.read()

        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), img)

        image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data= img)
        detection_result = detector.detect(image2)
        # STEP 5: Process the classification result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(image2.numpy_view(), detection_result)
        cv2.imshow('Live', annotated_image)
        cv2.waitKey(25)
         # This line will save the imges into the data folder
        
        counter += 1
    
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
