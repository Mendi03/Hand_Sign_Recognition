import cv2
import os

#from update_dataset import draw_rectangle

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

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26 # Number of hand signs to be detected, 26 is for each letter in the alphabet
dataset_size = 5 # Number of images to be taken

cap = cv2.VideoCapture(0)

# Loop for creating different folders containing the respective hand sign. Folder '0' is 'A' and folder '25' is 'Z'

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    while True:
        success, img = cap.read()
        cv2.putText(img, 
                    'Ready? Press "Q" ! :)', 
                    (100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.3, 
                    (0, 255, 0), 
                    3,
                    cv2.LINE_AA)
        
        draw_rectangle(img)
        cv2.imshow('live', img)
        #cv2.waitKey(500)

        #if cv2.waitKey(20) & 0xFF==ord('d'):
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0

    while(counter < dataset_size):
        ret, img = cap.read()
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), img) # This line will save the imges into the data folder
        draw_rectangle(img)
        cv2.imshow('Live', img)
        counter += 1
    
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
