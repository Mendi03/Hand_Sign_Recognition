import cv2
import os

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 500

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    while True:
        success, img = cap.read()
        cv2.putText(img, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        
        cv2.imshow('live', img)
        #cv2.waitKey(500)

        #if cv2.waitKey(20) & 0xFF==ord('d'):
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0

    while(counter < dataset_size):
        ret, img = cap.read()
        cv2.imshow('Live', img)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), img) # This line will save the imges into the data folder
        
        counter += 1
    
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
