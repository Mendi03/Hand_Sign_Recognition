import cv2
import os

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
    

#cap = cv2.VideoCapture(0)

No_of_frames = 195

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

DATA_DIR = '.\data'
good_dir = os.path.join(DATA_DIR, folder)


No_files = 0
# Iterate directory

# count number of files in specified folder
for file in os.listdir(good_dir):
    # check if current path is a file
    if os.path.isfile(os.path.join(good_dir, file)):
        No_files += 1
print('Current file count:', No_files)

print('Loading...')

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
    
    draw_rectangle(img)
    cv2.imshow('live', img)
    #cv2.waitKey(500)

    #if cv2.waitKey(20) & 0xFF==ord('d'):
    if cv2.waitKey(25) == ord('q'):
        break



new_total = No_of_frames + No_files

current_num = No_files

while(current_num < new_total):
    ret, img = cap.read()
    
    cv2.waitKey(25)
    cv2.imwrite(os.path.join(good_dir, '{}.jpg'.format(current_num)), img) # This line will save the imges into the data folder, it uses format method (Note: Maybe make it fstring?) to give correct name to the new captured frame
    draw_rectangle(img)
    cv2.imshow('Live', img)
    current_num += 1

cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()

