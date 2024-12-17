import cv2

# Load an image
image = cv2.imread('./test.jpg')

# Display the image
cv2.imshow('meeeee', image)
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close all OpenCV windows