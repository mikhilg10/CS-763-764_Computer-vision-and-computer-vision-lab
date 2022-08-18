from typing import DefaultDict
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from collections import defaultdict


def intruder_detector(intruder, db_image,n_points = 300):

    intruder = cv2.cvtColor(intruder, cv2.COLOR_BGR2GRAY)
    db_image = cv2.cvtColor(db_image, cv2.COLOR_BGR2GRAY)

    #resize images to same size
    points = (250,250)
    intruder = cv2.resize(intruder, points, interpolation= cv2.INTER_LINEAR)
    db_image = cv2.resize(db_image, points, interpolation= cv2.INTER_LINEAR)

    descriptor = cv2.SIFT_create()


    # Initiate ORB detector
    # if kp == 'fast':
        # detector = cv2.ORB_create(n_points)
    # elif kp == 'dog':
    detector = cv2.SIFT_create(n_points)
    interest_points_source = detector.detect(intruder,None)
    interest_points_target = detector.detect(db_image,None)
    _,descriptor_points_source = descriptor.compute(intruder,interest_points_source)
    _,descriptor_points_target = descriptor.compute(db_image,interest_points_target)
    
    Matcher = cv2.BFMatcher()
    # n_matches = Matcher.match(descriptor_points_source,descriptor_points_target)
    # sorted_n_matches = sorted(n_matches, key=lambda item: item.distance)
    n_matches = Matcher.knnMatch(descriptor_points_source,descriptor_points_target, k=2)
    # Apply Lowe's ratio test
    good_matches = []
    for m,n in n_matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])
        
    return good_matches


"""
part B qn 2------------------------------------------------------------------------
"""
folder = glob.glob("../replicate/*.jpg")
image_intruder = cv2.imread("../replicate/capturedImages/mask_low.jpg")
no_of_matches = []
match_dict = defaultdict(float)
n_points = 300
for image_path in folder:
    # Load our image template, this is our reference image
    image_db = cv2.imread(image_path) 
    
    # Get number of SIFT matches
    matches = intruder_detector(image_db, image_intruder,n_points)

    no_of_matches.append(len(matches)*1.0)
    # match_dict[image_path[13:]] = len(matches)*1.0
  
no_of_matches = np.round(np.array(no_of_matches)/(n_points),3)

threshold = 0.031
positives = [x for x in no_of_matches if x>=threshold]
# print(no_of_matches)
# print(positives)

"""
part B qn 3--------------------------------------------
"""
folder_db = glob.glob("../replicate/*.jpg")
folder_intruder =  glob.glob("../replicate/capturedImages/MIDDLE/*.jpg")
no_of_matches = []
match_dict = defaultdict(float)
threshold = 0.53
radius = [0,2,4,7,8,9,'inf']
x = []
i = 0
for index, intruder_path in enumerate(folder_intruder):
    
    image_intruder = cv2.imread(intruder_path)
    no_of_matches = []
    for image_path in folder_db:
        image_db = cv2.imread(image_path) 
        # Get number of SIFT matches
        matches = intruder_detector(image_db, image_intruder)
        no_of_matches.append(len(matches)*1.0)
        # match_dict[image_path[13:]] = len(matches)*1.0

    no_of_matches = np.array(no_of_matches)/300

    if no_of_matches[-1] >= 0.031:
        i += 1
        # print("circle radius:",radius[index])
        if i == 1:
            plt.imsave("../replicate/output/maskMiddleBest"+str(radius[index])+".jpg",image_intruder[:,:,[2,1,0]])
        x.append((radius[index],round(no_of_matches[-1],3)))
        

with open('../replicate/output/retScores.txt', 'w') as f:
    f.write('Radius')
    f.write('\t\t')
    f.write('Retrieval Score')
    for line in x:
        f.write('\n')
        f.write(str(line[0]))
        f.write('\t\t')
        f.write(str(line[1]))
        


