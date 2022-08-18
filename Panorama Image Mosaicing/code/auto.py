import cv2
import argparse
import os
from ransac import customRansac
from utils.common_scripts import getPoints, externalHomography, findDescriptorAndKeypoint, findMatches, displayMatching, stitchAndDisplay
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path',type=str, help="Path to two image directory")
    parser.add_argument('--mode', type=str, default='custom-ransac', help="Has two option custom-ransac for our ransac; and auto-ransac for opencv implementation")
    parser.add_argument('--normalize',type=int, help="normalization value", default=0)
    parser.add_argument('--error-type', type=str, \
        help="distance to measure the error in custom ransac, 2 options supported: Euclidean, Symmetric", 
        default="Euclidean")
    parser.add_argument('--adaptive', action='store_true', \
        help="for adaptive ransac, no need to give any value just --adaptive will do, valid only for mode=custom-ransac")
    parser.add_argument('--cropify', action="store_true",\
        help="for doing cropify after images , just use --cropify"
    )

    args = parser.parse_args()

    imgDirPath = args.path
    imageNames = []
    for file in os.listdir(imgDirPath):
        if os.path.isfile(os.path.join(imgDirPath, file)):    
            imageNames.append(os.path.join(imgDirPath, file))

    if len(imageNames)!=2:
        raise ValueError("Directory should have exactly 2 images")

    img1 = cv2.imread(imageNames[0])
    img2 = cv2.imread(imageNames[1])
    kp1, des1 = findDescriptorAndKeypoint(img1)
    kp2, des2 = findDescriptorAndKeypoint(img2)
    matches = findMatches(des1, des2)

    src_pts, dst_pts, Norm_s, Norm_d = getPoints(kp1,kp2,matches, normalize=args.normalize)

    
    if args.mode == "auto-ransac":
        if args.normalize:
            threshold = 0.05
        else:
            threshold = 1
        H, mask = externalHomography(src_pts,dst_pts, useCVRansac=True, threshold=threshold)
        inliers = mask.ravel() == 1
    else: # custom-ransac
        if args.normalize:
            threshold = 0.01
        else:
            threshold = 0.5
        inliers = customRansac(src_pts, dst_pts, threshold=threshold, adaptive=args.adaptive, error_type=args.error_type)
        if inliers is None:
            raise ValueError("Inliers couldn't be found, try increasing the threshold")
        H, mask = externalHomography(src_pts[inliers],dst_pts[inliers], useCVRansac=False, threshold=threshold)

    print("Total features points: {}\t Inliers found:{}\t Points Pruned(%): {:.4f}".format(len(inliers), sum(inliers), 1-(sum(inliers)/len(inliers))))

    res = stitchAndDisplay(img1, img2, H, Norm1=Norm_s, Norm2=Norm_d, windowName="Automatic Correspondence Stitching", useCropify=args.cropify)
    # cv2.imwrite("../results/lake.jpg",res)


