import cv2
import os
import numpy as np
import copy
from utils.common_scripts import stitchAndDisplay, undoNormalization
import argparse

def getTransformedPoints(H,pts):
    '''
    Transform source points so as to compare them with destination points to get the error between them
    '''
    src = np.c_[ pts, np.ones([pts.shape[0]]) ]
    dst = np.matmul(src,H.T)
    eps = 1e-20
    dst[dst[:, -1] == 0, -1] = eps
    dst /= dst[:, -1:]
    return dst[:,:-1]

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="path to directory containing two images")
parser.add_argument('--normalize', type=int, help="Whether to normalize or not")
args = parser.parse_args()

imgDirPath = args.path
imageNames = []
for file in os.listdir(imgDirPath):
    if os.path.isfile(os.path.join(imgDirPath, file)):    
        imageNames.append(os.path.join(imgDirPath, file))

if len(imageNames)!=2:
    raise ValueError("Directory should have exactly 2 images")

img1 = cv2.imread(imageNames[0])  # Reading image 1
img2 = cv2.imread(imageNames[1]) # Reading image 1
cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
cv2.moveWindow('image1',60,50)
cv2.resizeWindow('image1',600,600)

cv2.namedWindow('image2',cv2.WINDOW_NORMAL)
cv2.moveWindow('image2',650,50)
cv2.resizeWindow('image2',600,600)

img3=copy.deepcopy(img1)
img4=copy.deepcopy(img2)
#Let's detect points on images using mouseclick
pts_img1=[]
pts_img2=[]

def correspondences1(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        pts_img1.append((x,y))
        cv2.circle(img1,(x,y),2,(0,0,255),-1)

def correspondences2(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        pts_img2.append((x,y))
        cv2.circle(img2,(x,y),2,(0,0,255),-1)


while(1):
	cv2.setMouseCallback('image1',correspondences1)
	cv2.setMouseCallback('image2',correspondences2)
	cv2.imshow('image1',img1)
	cv2.imshow('image2',img2)
	if cv2.waitKey(20) & 0xFF == 27:
		break
cv2.destroyAllWindows()


def normalizePoints(points):
    '''
    Assuming input is 2D points
    '''
    points=np.float32(points)
    n = points.shape[0]
    xAvg, yAvg = np.mean(points[:,0]), np.mean(points[:,1])

    xy_norm =  np.sqrt(np.sum((points[:,0]-xAvg)**2 + (points[:,1]-yAvg)**2 )/(2*n))

    diag = 1/xy_norm
    m13 = -xAvg/xy_norm
    m23 = -yAvg/xy_norm

    norm_matrix = np.array([
        [diag, 0, m13],
        [0, diag, m23],
        [0, 0, 1]
    ])
    

    pts = np.c_[ points, np.ones([points.shape[0]]) ]
    norm_pts = pts@norm_matrix.T
    return norm_pts[:,:-1], norm_matrix

row = img2.shape[0] + img1.shape[0]
col = img2.shape[1] + img2.shape[1]

if (args.normalize==0):
    pts_img1=np.float32(pts_img1)
    pts_img2=np.float32(pts_img2)
    homography,status = cv2.findHomography(pts_img1,pts_img2)
    result = stitchAndDisplay(img3, img4, homography, useCropify=True)
    cv2.imwrite('../results/Nonorm_result.jpg', result)

    # symmetric error:
    H = homography.copy()
    distance = np.sqrt(np.sum((getTransformedPoints(H,pts_img1) - pts_img2)**2, axis=1) + \
            np.sum((pts_img1 - getTransformedPoints(np.linalg.inv(H),pts_img2))**2, axis=1))
    print("Symmetric Error : ",np.sum(distance))


elif(args.normalize==1):
    pts_img1, N1=normalizePoints(pts_img1)
    pts_img2, N2=(normalizePoints(pts_img2))

    homography,status = cv2.findHomography(pts_img1,pts_img2)
    H= np.linalg.inv(N2)@homography@N1
    H/=H[-1,-1]
    result = stitchAndDisplay(img3, img4, H, useCropify=True)
    cv2.imwrite('../results/norm_result.jpg', result)
    distance = np.sqrt(np.sum((getTransformedPoints(H,undoNormalization(pts_img1, N1)) - undoNormalization(pts_img2, N2))**2, axis=1) + \
            np.sum((undoNormalization(pts_img1, N1) - getTransformedPoints(np.linalg.inv(H),undoNormalization(pts_img2, N2)))**2, axis=1))
    print("Symmetric Error : ",np.sum(distance))


while True:

	cv2.imshow("result",result)
	if cv2.waitKey(0) & 0xFF == 27:
		break
cv2.destroyAllWindows()


'''
LET'S CALCULATE ERROR FOR NORMALIZED AND UNNORMALIZED MATRIX
'''
# img5= cv2.imread("../results/manual_norm_campus.jpg")
# img5=img5/np.max(img5)
# img6= cv2.imread("../results/manual_no_norm_campus.jpg")
# img6=img6/np.max(img6)


# h,w,_= img5.shape
# error= np.sqrt((np.sum((img5-img6)**2))/(h*w))
# # error= np.abs(img5-img6)/img5
# print("RMSE: ",error)



