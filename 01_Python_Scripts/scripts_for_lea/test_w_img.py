import cv2
import pickle
import numpy as np

# Load an image
image = cv2.imread('./test.jpg')

# Display the image
cv2.imshow('meeeee', image)
model_dict = pickle.load(open('./model.p', 'rb'))

model = model_dict['model']

prediction = model.predict([np.asarray(image)])

print(prediction)
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close all OpenCV windows