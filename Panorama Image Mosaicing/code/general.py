
import cv2
import fnmatch
import argparse
import os
import shutil
import numpy as np

from ransac import customRansac
from utils.common_scripts import findDescriptorAndKeypoint, findMatches, getPoints, externalHomography, displayMatching, stitchAndDisplay, undoNormalization


def startBFS(graph,indx = 0):
    '''
    BFS algorithm to generate BFS traversal and tree
    Input:
        graph: adjacency list like structure
        indx: node to start the BFS from.

    Returns:
        visited: Order of nodes (BFS Traversal)
        tree: BFS Tree with level, parent information of each node
    '''
    visited = []
    queue = []
    # traversal = []
    def bfs(graph, node):
        visited.append(node)
        queue.append(node)
        tree = {node:{"parent":None, "level":0}}
        while queue:
            s = queue.pop(0) 
            for neighbour in graph[s]:
                if neighbour not in visited:
                    tree[neighbour] = {"parent":s, "level":tree[s]["level"]+1}
                    visited.append(neighbour)
                    queue.append(neighbour)
        return tree

    tree = bfs(graph,indx)
    if len(visited) < len(graph):
        return None, None
    return visited, tree




def getHomography(img1, img2, mode="custom-ransac", adaptive=True, normalizePoints=True):
    '''
    Find homography between two images
    Input:
        img1: Image 1 (source)
        img2: Image 2 (destination)
        mode: Default custom-ransac, otherwise openCV Ransac supported
        adaptive: Boolean variable runs adaptive custom-ransac (default True)
        normalizePoints: Boolean variable to normalize points or not (default true)
    '''
    kp1, des1 = findDescriptorAndKeypoint(img1)
    kp2, des2 = findDescriptorAndKeypoint(img2)
    matches = findMatches(des1, des2)
    src_pts, dst_pts, Norm_s, Norm_d = getPoints(kp1,kp2,matches, normalize=normalizePoints)


    if mode == "custom-ransac":
        if normalizePoints:
            threshold = 0.01
        else:
            threshold = 0.5
        inliers = customRansac(src_pts, dst_pts, threshold=threshold, adaptive=adaptive)
        if inliers is None:
            return None, np.array([0]), None, None
            raise ValueError("Inliers couldn't be found, try increasing the threshold")
        H, mask = externalHomography(src_pts[inliers],dst_pts[inliers], useCVRansac=False,threshold=threshold)
    else: # auto-ransac
        if normalizePoints:
            threshold = 0.05
        else:
            threshold = 1
        H, mask = externalHomography(src_pts,dst_pts, useCVRansac=True, threshold=threshold)
        inliers = mask.ravel() == 1

    # displayMatching(undoNormalization(src_pts,Norm_s),undoNormalization(dst_pts, Norm_d),inliers,img1,img2)
    return H, inliers, Norm_s, Norm_d

def imageDependencyGraph(imageNames,ref_idx, mode="custom-ransac", adaptive=True, inlier_threshold=0.5, normalizePoints=True):
    '''
    Makes adjacency graph using inlier ratio. 
    For each node, a node is connected if inlier ration between them is greater than given inlier threshold
    For each node, list of adjacent node is sorted from highest similarity to lowest
    Input:
        imageNames: list of image names
        ref_idx: index of reference image name in imageNames list
        mode: Supports custom-ransac (default), otherwise openCV ransac is used
        adaptive: Sets apative custom-ransac (default true)
        inlier_threshold: used to classify if two images are similar on not on the basis of inlier found, 
                if number of inliers/ total points > inlier_threshold, then the images are similar otherwise not. (default 0.5)
        normalizePoints: Boolean variable to normalize points or not (default true)
    Returns:
        imgOrder: BFS traversal
        tree: BFS Tree
    '''
    n = len(imageNames)
    adjGraph = {i:[] for i in range(n)}
    print("Generating image dependency graph...")
    for i in range(n):
        img1 = cv2.imread(imageNames[i])
        print("Finding similarity with image {}".format(imageNames[i]))
        for j in range(i+1,n):
            img2 = cv2.imread(imageNames[j])
            _ , inliers, _, _ = getHomography(img1,img2, mode=mode, adaptive=adaptive, normalizePoints=normalizePoints)\
            
            # images are similar if they have more inlier percentage than given threshold
            inlier_ratio = sum(inliers)/len(inliers)
            if  inlier_ratio > inlier_threshold:
                adjGraph[i].append((j, inlier_ratio))
                adjGraph[j].append((i, inlier_ratio))
    
    # sorting similar images from highest to lowest inlier_ratio
    for k,v in adjGraph.items():
        sorted_v = sorted(v, key=lambda x: -x[1])
        adjGraph[k] = [n[0] for n in sorted_v]

    imgOrder, tree = startBFS(adjGraph, ref_idx)
    if imgOrder is None:
        raise ValueError("Some images are uncorrelated, cannot stitch. Please check set of images, keypoints, Decrease inlier_threshold")
    return imgOrder, tree




def stitchAndMerge(imageNames, img_ref_indx, imgs_indx_list):
    '''
    Stitches multiples images one-by-one with reference image
    Input: 
        imageNames: list of names of images
        img_ref_indx: index of image in imageNames that will be used reference image
        imgs_indx_list: indices of images in imageNames to be stitched with reference image
    Returns:
        name of stitched image/path: (str) saves stitched image temporarily
    ''' 
    # print(imageNames, imgs_indx_list)
    img_ref = cv2.imread(imageNames[img_ref_indx]) # reference image will be at first
    print("Reference Image: {}".format(imageNames[img_ref_indx]))
    for idx in imgs_indx_list:
        print("Currently Stitching Image: ",imageNames[idx])
        img = cv2.imread(imageNames[idx])

        H, _, Norm_s, Norm_d = getHomography(img, img_ref, mode=args.mode, adaptive=args.no_adaptive, normalizePoints=args.normalize) 
        img_ref = stitchAndDisplay(img, img_ref, H, Norm1=Norm_s, Norm2=Norm_d, windowName="img{}".format(imageNames[idx]))

    stitched_image_name = "./tmp/stitched_{}{}.jpg".format(img_ref_indx,"".join(map(str,imgs_indx_list)))
    # img_ref = cv2.resize(img_ref, None, fx=0.5, fy=0.5)

    cv2.imwrite(stitched_image_name, img_ref)
    return stitched_image_name


def built_bottom_up(tree, imageNames):
    '''
    Using tree generated from BFS, stitch in bottom-up fashion
    Input:
        tree: BFS tree with each node having parent and level information
        imageNames: list of names of images to be stitched together
    Returns:
        Result image name/path: (str) Root of tree after being stitched completely has name of resultant file
    '''
    level_bins = {}
    max_level = 0
    for k,v in tree.items():
        l = v["level"]
        if l in level_bins.keys(): level_bins[l].append(k)
        else: level_bins[l] = [k]
        max_level = max(max_level, l)
        
    level = max_level
    
    for level in range(max_level,0,-1):
        leaves = level_bins[level]
        parent= {}
        for leaf in leaves:
            if tree[leaf]["parent"] in parent.keys():
                parent[tree[leaf]["parent"]].append(leaf)
            else:
                parent[tree[leaf]["parent"]] = [leaf]
        print(parent)
        for p in parent.keys():
            imageNames[p] = stitchAndMerge(imageNames, p, parent[p])

    return imageNames[level_bins[0][0]]
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path',type=str, help="Path to two image directory")
    parser.add_argument('--mode', type=str, default='custom-ransac', help="Has two option custom-ransac for our ransac; and auto-ransac for opencv implementation")
    parser.add_argument('--normalize',type=int, help="normalization value", default=0)
    parser.add_argument('--idx', type=int, \
        help="reference image index around with other images will be stitched, kindly number the images sequentially and use that number")
    parser.add_argument('--no-adaptive', action='store_false', \
        help="default is adaptive ransac, no need to give any value just --no-adaptive will do, valid only for mode=custom-ransac")
    parser.add_argument("--order", nargs='+', type=int, \
        help="""order of stitching of images (when image names are sorted), 
        we recommend to use this option if you know the order of the images names after being sorted,
        Also Reference image index has to come first
        eg, --order 1 3 2 here 1 3 2 implies after sorting images names stitch 1(Ref. Image), then 3, then 2
        """,
        default=[])
    parser.add_argument('--inlier-threshold',type=float,\
        default=0.8,
        help="used to classify if two images are similar on not on the basis of inlier found, if number of inliers/ total points > inlier_threshold, then the images are similar otherwise not. (default 0.5)")


    # sample running cmd:
    # python general.py ../data/mountain/ --idx 2 --mode custom-ransac --normalize 1 
    # python general.py ../data/mountain/ --idx 2 --mode auto-ransac --normalize 1 --order 2 1 3 4 5
    # python general.py ../data/mountain --idx 2 --mode custom-ransac --normalize 1--order 2 4 5 1 3
    # python general.py ../data/hall/ --idx 3 --mode custom-ransac --normalize 1

    args = parser.parse_args()

    imgDirPath = args.path
    imageNames = []
    imgRef = ""
    imgRefSeen = 0
    for file in os.listdir(imgDirPath):
        if os.path.isfile(os.path.join(imgDirPath, file)):   
            if fnmatch.fnmatch(file, '{}.*'.format(args.idx)) :
                if not imgRefSeen:
                    imgRef = os.path.join(imgDirPath, file)
                    imgRefSeen = 1
                else:
                    raise ValueError("Two or more images with same index found")
            imageNames.append(os.path.join(imgDirPath, file))

    if (len(imageNames)<3 or len(imageNames)>5):
        raise ValueError("Directory should have atleast 3 and atmost 5 images")
    if imgRef == "":
        raise ValueError("{}.(png|jpg|or any format) not found, Please see argparser help for idx parameter".format(args.idx))

    imageNames = sorted(imageNames)
    args.idx = imageNames.index(imgRef)
    print("Images:",imageNames)
    print("New Ref Index:",args.idx)
    
    tmp_dir = "./tmp"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    if args.order != []:
        # if image order is provided (we suggest to refrain from this as out-)
        imgOrder = np.array(args.order) - 1 
        print(imgOrder)
        final_stitched_img_name = stitchAndMerge(imageNames, imgOrder[0], imgOrder[1:])
        final_stitched_img = cv2.imread(final_stitched_img_name)
        cv2.namedWindow("Final Stitched Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Final Stitched Image", final_stitched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        if args.normalize:
            inlier_threshold=0.7
        else:
            inlier_threshold=0.5
        imgOrder, tree = imageDependencyGraph(imageNames, args.idx, mode=args.mode, adaptive=args.no_adaptive, inlier_threshold=inlier_threshold, normalizePoints=args.normalize)
        print("Image stitching Order: ",imgOrder)


        print("\nFinding order of stitching using BFS Tree in bottom-up fashion:")
        final_stitched_img_name = built_bottom_up(tree, imageNames)

        final_stitched_img = cv2.imread(final_stitched_img_name)
        cv2.namedWindow("Final Stitched Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Final Stitched Image", final_stitched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # cv2.imwrite("../results/general_mountain_img_2_ref_img.jpg",final_stitched_img)

    if os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)