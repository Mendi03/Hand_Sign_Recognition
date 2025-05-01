import cv2
import os

#cap = cv2.VideoCapture(0)

# Choose folder to add images to

folder1 = input("Enter folder to update:")

folder_name = int(folder1)

print(folder_name)

if folder_name < 0 or folder_name > 25:
    print("invalid folder name")
    quit()

else:
    folder = str(folder_name)
    print("Folder chosen is: " + folder)

# print(os.getcwd())
# print(os.listdir())

DATA_DIR = './data'

print('Collecting data for folder: ' + folder)

cap = cv2.VideoCapture(0)

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
    
    cv2.imshow('live', img)
    #cv2.waitKey(500)

    #if cv2.waitKey(20) & 0xFF==ord('d'):
    if cv2.waitKey(25) == ord('q'):
        break

No_of_frames = 50

while(counter < No_of_frames):
    ret, img = cap.read()
    cv2.imshow('Live', img)
    cv2.waitKey(25)
    cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), img) # This line will save the imges into the data folder, it uses format method instead of fstring to give correct name to the image
    
    counter += 1

cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
