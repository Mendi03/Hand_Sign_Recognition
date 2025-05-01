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

DATA_DIR = '.\data'

good_dir = os.path.join(DATA_DIR, folder)

print(os.getcwd())

print(os.listdir(good_dir))

#data\0\0.jpg

No_files = 0
# Iterate directory


print(good_dir)

# count number of files in specified folder
for file in os.listdir(good_dir):
    # check if current path is a file
    if os.path.isfile(os.path.join(good_dir, file)):
        No_files += 1
print('File count:', No_files)

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

No_of_frames = 10

new_total = No_of_frames + No_files

current_num = No_files

while(current_num < new_total):
    ret, img = cap.read()
    cv2.imshow('Live', img)
    cv2.waitKey(25)
    cv2.imwrite(os.path.join(good_dir, '{}.jpg'.format(current_num)), img) # This line will save the imges into the data folder, it uses format method (Maybe make it fstring) to give correct name to the new captured frame
    
    current_num += 1

cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
