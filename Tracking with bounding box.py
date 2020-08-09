import cv2
import numpy as np

cap= cv2.VideoCapture("walk.mp4")

ret, frame1= cap.read()
ret, frame2= cap.read()

while cap.isOpened():

    diff= cv2.absdiff(frame1, frame2)# absolute difference between frame1 and frame2

    gray= cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    #blur grayscale
    blur= cv2.GaussianBlur(gray, (5,5), 0)
    _,thresh= cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)

    #dilate the thresholded frames to fill-in all the holes
    dilate= cv2.dilate(thresh, None, iterations=3)

    # fnding contour on dilated frame
    contour, hir = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    #iterate over all counter
    for cont in contour:
        (x,y,w,h)=cv2.boundingRect(cont)# bounding box(Rectangle)


        if cv2.contourArea(cont)<30000:
            continue
        else:
            cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0), 3)
            cv2.putText(frame1, "Movement", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)




    #drawing contour
    # cv2.drawContours(frame1, contour, -1, (0,255,0), 3)


    cv2.imshow('feed', frame1)
    frame1= frame2
    ret, frame2= cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()