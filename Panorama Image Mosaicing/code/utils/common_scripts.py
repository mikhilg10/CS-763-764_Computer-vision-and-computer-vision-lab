
import cv2
import numpy as np
from crop import cropify

def normalizePoints(points):
    '''
    Assuming input is 2D points
    Normalizing Points as described in the task
    '''
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

def undoNormalization(points, Norm_matrix):
    '''
    Denormalize the points using normalization matrix
    '''
    pts = np.c_[ points, np.ones([points.shape[0]]) ]
    pts = pts@np.linalg.inv(Norm_matrix.T)
    return pts[:,:-1]

def getPoints(kp1,kp2,matches, normalize=False):
    '''
    Input: keypoints from Image1, Image2 and matches
    Returns: source and destination points from matches
    '''
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,2)

    if normalize:
        src_pts, Norm_s = normalizePoints(src_pts)
        dst_pts, Norm_d = normalizePoints(dst_pts)
        return src_pts, dst_pts, Norm_s, Norm_d
    return src_pts, dst_pts, np.eye(3), np.eye(3)


def externalHomography(src_pts, dst_pts, useCVRansac=False, threshold=1.0):
    '''
    Do homography on top of source and destination points
    '''
    H, mask = None, None
    if useCVRansac:
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,threshold)
    else: # custom ransac already applied
        H, mask = cv2.findHomography(src_pts, dst_pts)
    return H, mask


def findDescriptorAndKeypoint(img):
    '''
    Input: Image
    Description: Using SIFT as detector and descriptor as it is widely used and it's performance is still hard to beat on various fronts
    Returns: keypoint, descriptor
    '''
    sift = cv2.xfeatures2d.SIFT_create(nfeatures = 1024, nOctaveLayers = 4, contrastThreshold=0.03)
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def knnMatcher(desc1,desc2):
    '''
    Input: Descriptors of 2 images
    Description: Knn based matcher used to extend more keypoints
    Returns: Matches
    '''
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = matcher.knnMatch(desc1, desc2, 2)
    pruned_matches = []
    n1 = 0.75
    for m,n in matches:
        if m.distance < n1 * n.distance :
            pruned_matches.append(m)
    return pruned_matches

def bruteForceMatcher(des1, des2):
    '''
    Input: Descriptors of 2 images
    Description: Brute force based matcher used to get keypoints
    Returns: Matches
    '''
    brute_force = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    return brute_force.match(des1, des2)

def findMatches(des1, des2):
    '''
    Input: Descriptors of 2 images
    Description: performs Knn and Brute force matching
    Returns: Matches
    '''
    matches = []
    matches = bruteForceMatcher(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)[:int(.1*len(matches))] # top 10% points selected
    # matches_knn = knnMatcher(des1, des2)
    # matches.extend(sorted(matches_knn, key=lambda x:x.distance)[:int(.1*len(matches_knn))])
    return matches

def displayMatching(src_pts, dst_pts, inliers,img1,img2):
    '''
    To display matches
    '''
    inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
    inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
    placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(np.sum(inliers))]
    matchedImage = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)

    cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
    cv2.imshow('Matches', matchedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def crop(image):
    '''
    crops extra black space from an image
    '''
    y_nonzero, x_nonzero, _ = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

def pointComparator(pt1, pt2):
    # check x co-ord
    if pt1[0]<pt2[0]: 
        return pt1
    elif pt2[0]<pt1[0]:
        return pt2
    else:
        # x is equal
        # now check y
        if pt1[1]<pt2[1]:
            return pt1
        elif pt2[1]<pt1[1]:
            return pt2
        # both equal 
        return pt1

def selectCorner(img1_pts, img2_pts):
    corners = []
    bl_m = np.array([
        [1,0],
        [0,-1]
    ])
    br_m = np.array([
        [-1,0],
        [0,-1]
    ])
    tr_m = np.array([
        [-1,0],
        [0,1]
    ])
    tl = pointComparator(img1_pts[0],img2_pts[0])

    bl = pointComparator(img1_pts[1]@bl_m,img2_pts[1]@bl_m)

    br = pointComparator(img1_pts[2]@br_m,img2_pts[2]@br_m)

    tr = pointComparator(img1_pts[3]@tr_m,img2_pts[3]@tr_m)
    return np.array([tl,bl@np.linalg.inv(bl_m),br@np.linalg.inv(br_m),tr@np.linalg.inv(tr_m)])


def stitchAndDisplay(img1, img2, H, Norm1=np.eye(3), Norm2=np.eye(3), windowName="Stitched", useCropify=False):
    '''
    warp img1 to img2 with homograph H
    Inputs: 
        img1: image 1
        img2: image 2, 
        Norm1: matrix used for normalization (if any) of image 1 points, if no normalization done then don't pass value for this attr
        Norm1: matrix used for normalization (if any) of image 2 points, if no normalization done then don't pass value for this attr
        windowName: To give stitched window a name
    Returns:
        stitched image -> img1 is stitched to img2
    '''

    assert isinstance(Norm1, np.ndarray), "Normalization matrix Norm1 should be a 3x3 numpy array"
    assert isinstance(Norm2, np.ndarray), "Normalization matrix Norm2 should be a 3x3 numpy array"

    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
   
    
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)

    '''
    Denormalization: recover Homography H from H_norm if points were normalized before homography
    '''
    H = np.linalg.inv(Norm2)@H@Norm1
    H /= H[-1,-1] # minor errors can creep in
    pts1_ = cv2.perspectiveTransform(pts1, H)

    pts = np.concatenate((pts1_, pts2), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() -0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() +0.5)

    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    result = cv2.warpPerspective(img1, Ht.dot(H), (xmax-xmin, ymax-ymin), flags=cv2.INTER_NEAREST)

    pt1 = pts1_.reshape(-1,2)
    pt2 = pts2.reshape(-1,2)
    pt1 = np.c_[ pt1, np.ones([pt1.shape[0]]) ] 
    pt2 = np.c_[ pt2, np.ones([pt2.shape[0]])] 
    pt1 = pt1@Ht.T
    pt2 = pt2@Ht.T
    corners = selectCorner(pt1[:,:-1], pt2[:,:-1])

    img2_placeholder = result[t[1]:h2+t[1],t[0]:w2+t[0]]
    mask = (img2 >=0) & (img2<=10)
    img2[mask] = img2_placeholder[mask]

    result[t[1]:h2+t[1],t[0]:w2+t[0]] = img2

    
    if useCropify:
        result = cropify(result, corners)
    result = crop(result)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.imshow(windowName, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return result
