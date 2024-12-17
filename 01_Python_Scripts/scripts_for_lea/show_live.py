import cv2

cap = cv2.VideoCapture(0)

count = 0

while True:
    success, img = cap.read()
    cv2.imshow('live', img)
    cv2.imwrite('./data/save%d.jpg' % count, img) # This line will save the imges into the data folder
    cv2.waitKey(500)
    count = count + 1

    #if cv2.waitKey(20) & 0xFF==ord('d'):
    if cv2.waitKey(1) == ord('d'):
        print(count)
        break

    elif(count == 100):
        break

cap.release()
cv2.destroyAllWindows()
