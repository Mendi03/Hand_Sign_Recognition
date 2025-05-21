from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
import os
import pickle

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

data = []
labels = []

def store_landmarks(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx] 

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        #print(hand_landmarks_proto)

        hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
        solutions.drawing_utils.draw_landmarks(annotated_image,hand_landmarks_proto,
                                               solutions.hands.HAND_CONNECTIONS,
                                               solutions.drawing_styles.get_default_hand_landmarks_style(),
                                               solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape

        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]

        both_coords = x_coordinates + y_coordinates
        
        data.append(both_coords)

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

DATA_DIR = './data'


for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        image2 = mp.Image.create_from_file(os.path.join(DATA_DIR, dir_, img_path))

        detection_result = detector.detect(image2)
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness

        detected_hands = len(hand_landmarks_list)

        # Checks if amount of hands detected in image are 1. Also skips images where hand were not detected.
        if len(hand_landmarks_list) == 1:
            store_landmarks(image2.numpy_view(), detection_result)
            labels.append(dir_)
        else:
            pass

print(len(data))
print(len(labels))

f = open('data2.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

