import pickle

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def draw_rectangle(image):
    size = image.shape

    top_left_x = int(size[1] / 2 - 100)
    top_left_y = int(size[0] / 2 - 100)

    bottom_right_x = int(size[1] / 2 + 100)
    bottom_right_y = int(size[0] / 2 + 100)

    cv2.rectangle(image,                                # Image to draw on
                    (top_left_x, top_left_y),           # upper left corner coordinates
                    (bottom_right_x, bottom_right_y),   # bottom right corner coordinates
                    (255,0,0),                          # BGR color
                    2)                                  # Line thickness
    
model_dict = pickle.load(open('./model2.p', 'rb')) # Error might occur in script if folder opened in vs code isn't the right one. In my case, the folder opened should be 01_Python_Scripts.
model = model_dict['model']

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2) # Need to specify 2 hands; if not specified, the script crashes. 
detector = vision.HandLandmarker.create_from_options(options)


def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Important info on Landmarks:

    There are 21 hand landmarks, each composed of x, y and z coordinates. The x and y coordinates are normalized
    to [0.0, 1.0] by the image width and height, respectively. The z coordinate represents the landmark depth, with 
    the depth at the wrist being the origin. The smaller the value, the closer the landmark is to the camera. The 
    magnitude of z uses roughly the same scale as x.

    Taken from: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
    """

    # Obtain all data from hand (Left or right, number of hands, hand lanmarks, etc.)

    MARGIN = 20  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green    sdvccx 

    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
                6: 'G',7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 
                12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 
                18: 'S',19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
                24: 'Y', 25: 'Z'}

    hand_landmarks_list = detection_result.hand_landmarks
    #handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        #handedness = handedness_list[idx] 

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

        hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
        solutions.drawing_utils.draw_landmarks(annotated_image,
                                               hand_landmarks_proto,
                                               solutions.hands.HAND_CONNECTIONS,
                                               solutions.drawing_styles.get_default_hand_landmarks_style(),
                                               solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape

        x_coordinates = [landmark.x for landmark in hand_landmarks] # Landmark.x 
        y_coordinates = [landmark.y for landmark in hand_landmarks]

        min_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) #- MARGIN

        # Detects the hand sign and stores predicted letter to 'predicted_character' variable
        both_coords = x_coordinates + y_coordinates
        prediction = model.predict([np.asarray(both_coords)])
        predicted_character = labels_dict[int(prediction[0])]

        #print(predicted_character)

        # Draw rectangle around hand
        min_y = int(min(y_coordinates) * height)
        box_x = int(max(x_coordinates) * width)
        box_y = int(max(y_coordinates) * height)


        # Square size for hand logic: 
        sq_width = box_x - min_x
        sq_height = box_y - min_y

        if sq_width > sq_height:
            ten_percent = int(sq_width * 0.3)

            MIN_X = min_x - ten_percent
            MAX_X = box_x + ten_percent

            extra = int((sq_width - sq_height) / 2)
            MIN_Y = min_y - extra - ten_percent
            MAX_Y = box_y + extra + ten_percent

        else:
            ten_percent = int(sq_height * 0.3)

            MIN_Y = min_y - ten_percent
            MAX_Y = box_y + ten_percent

            extra = int((sq_height - sq_width) / 2)
            MIN_X = min_x - extra - ten_percent
            MAX_X = box_x + extra + ten_percent
            
            


        #padding = 20
        
        cv2.rectangle(annotated_image, # Image to draw on
                    (MIN_X , MIN_Y ), # upper left corner coordinates
                    (MAX_X , MAX_Y ), # bottom right corner coordinates
                    (255,0,0), # BGR color
                    2) # Line thickness
        
                    
        # Draw predicted letter.
        cv2.putText(annotated_image, # Image to show
                    predicted_character, # Labeled letter
                    (min_x, text_y), 
                    cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, 
                    HANDEDNESS_TEXT_COLOR, 
                    FONT_THICKNESS, cv2.LINE_AA)
        
    return annotated_image

