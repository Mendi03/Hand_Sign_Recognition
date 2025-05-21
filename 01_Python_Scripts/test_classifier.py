import functions
import pickle

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


cap = cv2.VideoCapture(0)



base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2) # Need to specify 2 hands; if not specified, the script crashes. 
detector = vision.HandLandmarker.create_from_options(options)

while True:
    success, img = cap.read()
    cv2.waitKey(20)

    # Convert frames to SRGB
    image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data= img)

    # Obtain all data from hand (Left or right, number of hands, hand lanDmarks, etc.)
    detection_result = detector.detect(image2)

    # Create new frame with Labeled hand sign and drawn results 
    annotated_image = functions.draw_landmarks_on_image(image2.numpy_view(), detection_result)

    # Show new frame in live feed
    #draw_rectangle(annotated_image)
    cv2.imshow('image', annotated_image)

    # If 'q' is pressed end script
    if cv2.waitKey(1) == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
