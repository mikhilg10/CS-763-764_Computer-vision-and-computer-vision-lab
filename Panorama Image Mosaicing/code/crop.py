import cv2
import numpy as np


def cropify(img, pts1):
    w,h = 800, 600
    img = cv2.resize(img,(w,h))

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray,3)

    _, thresh = cv2.threshold(gray,1,255,0)
    _, contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=lambda contour:len(contour), reverse=True)
    roi = cv2.boundingRect(contours[0])

    
    pts1= pts1.astype(np.float32) #Distorted
    pts2= np.array([[0,0], [0,h], [w,h], [w,0]]).astype(np.float32) # Undistorted

    # row, col, _ = img1.shape
    img_new= img[roi[1]:roi[3], roi[0]:roi[2]]
    a= cv2.getPerspectiveTransform(pts1, pts2)
    undistorted = cv2.warpPerspective(img_new, a, (w, h))

    return undistorted