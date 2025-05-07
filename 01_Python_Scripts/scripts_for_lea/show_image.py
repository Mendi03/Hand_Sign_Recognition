import cv2

# Load an image
image = cv2.imread('scripts_for_lea/test.jpg')
image2 = cv2.imread('data/0/100.jpg')

size = image2.shape

img_rz = cv2.resize(image, (640,480))

print(image2.shape)
print(image.shape)
print(img_rz.shape)
print(type(size))


# Display the image
cv2.imshow('meeeee', img_rz)

print(type(size[0]))

top_left_x = int(size[1] / 2 - 100)
top_left_y = int(size[0] / 2 - 100)

bottom_right_x = int(size[1] / 2 + 100)
bottom_right_y = int(size[0] / 2 + 100)

print(top_left_x)
print(top_left_y)
print(bottom_right_x)
print(bottom_right_y)

print(bottom_right_x - top_left_x)
print(bottom_right_y - top_left_y)

cv2.rectangle(image2, # Image to draw on
                    (top_left_x, top_left_y), # upper left corner coordinates
                    (bottom_right_x, bottom_right_y), # bottom right corner coordinates
                    (255,0,0), # BGR color
                    2) # Line thickness

cv2.imshow('thingy', image2)

cv2.waitKey(0)  # Wait for a key press

cv2.destroyAllWindows()  # Close all OpenCV windows