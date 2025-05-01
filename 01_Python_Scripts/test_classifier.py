import pickle

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_dict = pickle.load(open('./model.p', 'rb')) # Error might occur in script if folder opened in vs code isn't the right one. In my case, the folder opened should be 01_Python_Scripts.
model = model_dict['model']

cap = cv2.VideoCapture(0)

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green    sdvccx 

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
               6: 'G',7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 
               18: 'S',19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
               24: 'Y', 25: 'Z'}

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
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

        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]

        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        both_coords = x_coordinates + y_coordinates
        
        prediction = model.predict([np.asarray(both_coords)])
        
        predicted_character = labels_dict[int(prediction[0])]

        #print(predicted_character)

        # Draw rectangle around hand
        min_y = int(min(y_coordinates) * height)
        box_x = int(max(x_coordinates) * width)
        box_y = int(max(y_coordinates) * height)

        padding = 20

        cv2.rectangle(annotated_image, (text_x - padding, min_y - padding), (box_x + padding, box_y + padding), (255,0,0), 2)

        # Draw predicted letter.
        cv2.putText(annotated_image, 
                    predicted_character,
                    (text_x, text_y), 
                    cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, 
                    HANDEDNESS_TEXT_COLOR, 
                    FONT_THICKNESS, cv2.LINE_AA)
        
    return annotated_image


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2) # Need to specify 2 hands; if not specified, the script crashes. 
detector = vision.HandLandmarker.create_from_options(options)

while True:
    success, img = cap.read()
    cv2.waitKey(20)

    # Convert frames to SRGB
    image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data= img)

    # Obtain all data from hand (Left or right, number of hands, hand lanmarks, etc.)
    detection_result = detector.detect(image2)

    # Create new frame with Labeled hand sign and drawn results 
    annotated_image = draw_landmarks_on_image(image2.numpy_view(), detection_result)

    # Show new frame in live feed
    cv2.imshow('image', annotated_image)

    # If 'q' is pressed end script
    if cv2.waitKey(1) == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
