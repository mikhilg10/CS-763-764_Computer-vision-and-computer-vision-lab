import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time


parser = argparse.ArgumentParser()
parser.add_argument("--trans",help="[view|scale|rot|light]")
parser.add_argument("--nm",help="Mention the number of matches",type=int,default=50)
args = parser.parse_args()
kp = args.kp
des = args.des
trans = args.trans
nm = args.nm  


if trans == 'view':
    img_s = plt.imread('../data/created/view_S.jpeg')
    img_t = plt.imread('../data/created/view_T.jpeg')
elif trans == 'scale':
    img_s = plt.imread('../data/created/scale_S.jpeg')
    img_t = plt.imread('../data/created/scale_T.jpeg')
elif trans == 'rot':
    img_s = plt.imread('../data/created/rot_S.jpeg')
    img_t = plt.imread('../data/created/rot_T.jpeg')
elif trans == 'light':
    img_s = plt.imread('../data/created/light_S.jpeg')
    img_t = plt.imread('../data/created/light_T.jpeg')

gray_s = cv.cvtColor(img_s, cv.COLOR_RGB2GRAY)
gray_t = cv.cvtColor(img_t,cv.COLOR_RGB2GRAY)


descriptor = cv.xfeatures2d.FREAK_create()


for n_points in [300,1000]:
    detector = cv.xfeatures2d.SURF_create(n_points)
    start = time.time()
    interest_points_source = detector.detect(gray_s,None)
    interest_points_target = detector.detect(gray_t,None)
    _,descriptor_points_source = descriptor.compute(gray_s,interest_points_source)
    _,descriptor_points_target = descriptor.compute(gray_t,interest_points_target)

    Matcher = cv.BFMatcher(cv.NORM_L2,crossCheck = True)
    n_matches = Matcher.match(descriptor_points_source,descriptor_points_target)
    sorted_n_matches = sorted(n_matches, key=lambda item: item.distance)
    result = cv.drawMatches(img_s, interest_points_source, img_t, interest_points_target, sorted_n_matches[0:nm], img_t, flags = 2)
 
    plt.imsave('../results/created/'+str(kp)+'_'+str(des)+'_'+str(trans)+'_'+str(n_points)+'.png',result)
    plt.imshow(result)
    plt.title(f'Number of interest_points: %d' % n_points)
    plt.show()
    

