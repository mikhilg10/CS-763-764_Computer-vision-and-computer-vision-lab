import cv2
import argparse
import numpy as np
from Utils.VideoUtils import VideoReader, VideoWriter
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
import time

def real_func(frame,index):

    """
    This function filters out only stable object features and return keypoints,descriptors
    """

    video_1 = VideoReader("../data", args.question, args.video)
    f_n = video_1.getFlow(index)
    f_n_mean = np.mean(f_n, axis=(0, 1))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_frame1 = gray
    height = current_frame1.shape[0]
    width = current_frame1.shape[1]

    current_frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for x in range(height):
        for y in range(width):
            l2_norm = np.sqrt(((f_n[x][y][0] - f_n_mean[0]) ** 2) + ((f_n[x][y][1] - f_n_mean[1]) ** 2))
            if l2_norm < 5:
                #print("sdc")
                #continue
                current_frame2[x][y] = 0
            else:
                current_frame1[x][y] = 0

    #cv2.imshow("gray frame1", current_frame1)
    #cv2.imshow("gray frame2", current_frame2)

    sift = cv2.SIFT_create()
    kp1,des1 = sift.detectAndCompute(current_frame1,None)
    img1 = cv2.drawKeypoints(frame,kp1,current_frame1,(0,255,0))
    kp2,des2 = sift.detectAndCompute(current_frame2,None)
    final_stable = cv2.drawKeypoints(img1,kp2,current_frame2,(0,0,255))

    kp_fi, des_fi = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),None)
    final = cv2.drawKeypoints(frame, kp_fi,cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) , (0, 255, 0))

    frame_out = cv2.hconcat([final, final_stable])
    path = get_video_path(args.question, args.subQuestion, args.video)
    output_video = VideoWriter(path)
    output_video.writeFrame(frame_out)
    cv2.imshow("Before and After", frame_out)
    cv2.waitKey(10)


    return kp1,des1


def get_transformation_mat(img1_colour, img2_colour):
    """
    This function takes two images and gives the homography matrix between them
    """
    sift = cv2.xfeatures2d.SIFT_create()

    # Loading Image and converting it into gray scale
    img1 = cv2.cvtColor(img1_colour, cv2.COLOR_BGR2GRAY)
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)


    # Loading Image and converting it into gray scale
    img2 = cv2.cvtColor(img2_colour, cv2.COLOR_BGR2GRAY)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)


    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)


    dist_threshold = 0.5
    good_matches = [m for m,n  in matches if m.distance < dist_threshold * n.distance]
    src_pts = np.float32([ keypoints_1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints_2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
   
    (transform, status) = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return transform
    
    
def get_transformation_mat_stable(img1_colour,img1_index, img2_colour, img2_index):

    """
    This function takes two images and gives the homography matrix between them of stable object features
    """
    keypoints_1, descriptors_1 = real_func(img1_colour,img1_index)
    keypoints_2, descriptors_2 = real_func(img2_colour,img2_index)
    
    

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)


    dist_threshold = 0.5
    good_matches = [m for m,n  in matches if m.distance < dist_threshold * n.distance]
    src_pts = np.float32([ keypoints_1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints_2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
   
    (transform, status) = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return transform
    

def get_translation_from_mat(transform):
    """
    This function outputs the dx dy and angle from the transform
    """
    # translation x
    dx = transform[0, 2]
    # translation y
    dy = transform[1, 2]
    # rotation
    da = np.arctan2(transform[1, 0], transform[0, 0])
    return [dx, dy, da]


def gaussian_filter_1d(size,sigma):
    """
    This function takes input the sigma and outputs the gaussian filter for smoothing purspose
    """
    filter_range = np.linspace(-int(size/2),int(size/2) + 1,size)
    gaussian_filter = [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2)) for x in filter_range]
    return np.array(gaussian_filter)


def gauss_smoothing(inp_array, sigma):
    """
    This function takes ouputs the gaussian filter
    """
    # Todo Implement this without library function
    output = gaussian_filter1d(inp_array, sigma, mode='mirror')
    # kernel = cv2.getGaussianKernel( 3, sigma )
    # kernel = gaussian_filter_1d(5, sigma)
    # output = cv2.filter2D(inp_array, -1, kernel, borderType = cv2.BORDER_REFLECT).T[0]

    return output



def smooth(trajectory, sigma):
    """
    This function outputs the smoothed trajectory after processing the x y and angle of the trajectory
    """
    smoothed_trajectory = np.copy(trajectory)

    smoothed_trajectory[:,0] = gauss_smoothing(trajectory[:,0], sigma) #For X
    smoothed_trajectory[:,1] = gauss_smoothing(trajectory[:,1], sigma) #For Y
    smoothed_trajectory[:,2] = gauss_smoothing(trajectory[:,2], sigma)  #For theta

    return smoothed_trajectory



def get_width_height(frame, H):
    """
    This function takes frame ad a input and gives dimensions as output
    """
    height, width = frame.shape[:2]
    corners = np.float32([[0,0],[0,height],[width,height],[width,0]]).reshape(-1,1,2)
    trans_corners = cv2.perspectiveTransform(corners, H)
    [xmin, ymin] = np.int32(trans_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int32(trans_corners.max(axis=0).ravel())
    width = xmax - xmin
    height = ymax - ymin
    return (width, height)


def get_frame_transformation(video_frames):
    """
    This function is used to get the transformation between frames for feature transform
    """
    prev_frame = video_frames[0]

    transforms = []

    for i in range(1, len(video_frames)-1):
        curr_frame = video_frames[i]
        mat = get_transformation_mat(prev_frame, curr_frame)
        trans = get_translation_from_mat(mat)
        transforms.append(trans)
        prev_frame = curr_frame
    return transforms
    
def get_frame_transformation_stable(video_frames):
    """
    This function is used to get the transformation between frames for feature transform of stable object faetures
    """
    prev_frame = video_frames[0]
    prev_frame_index = 0

    transforms = []

    for i in range(1, len(video_frames)-1):
        curr_frame = video_frames[i]
        curr_frame_index = i
        mat = get_transformation_mat_stable(prev_frame, prev_frame_index, curr_frame, curr_frame_index)
        trans = get_translation_from_mat(mat)
        transforms.append(trans)
        prev_frame = curr_frame
        prev_frame_index = curr_frame_index
    return transforms

def plot_trajectory_smoothing_diff(trajectory, smoothed_trajectory):
    """
    This function plots the original trajectory and smoothed trajectory
    """
    # filter = cv2.getGaussianKernel( trajectory[:,0].shape[0], 100 )
    # plt.plot(np.arange(len(filter)), filter, color ="green")
    plt.plot(np.arange(len(trajectory)), trajectory[:,1], color ="blue")
    plt.plot(np.arange(len(smoothed_trajectory)), smoothed_trajectory[:,1], color ="red")
    plt.show()

def get_transformation_matrix_from_affine_values(transform_values):
    """
    This function is used to get the transformation matrix
    """
    dx, dy, da = transform_values[0], transform_values[1], transform_values[2]

    # Reconstruct transformation matrix accordingly to new values
    m = np.zeros((3,3), np.float32)
    m[0,0] = np.cos(da)
    m[0,1] = -np.sin(da)
    m[1,0] = np.sin(da)
    m[1,1] = np.cos(da)
    m[0,2] = dx
    m[1,2] = dy
    m[2, 2] = 1
    return m

def get_video_path(q_no, subq_no, video_no):
    """
    This function gets the output path
    """
    path = f'../Output/{q_no}_{subq_no}'
    if not os.path.exists(path):
        os.makedirs(path)

    path = path + f'/{video_no}.avi'
    return path

def get_masking_video_path(q_no, subq_no, video_no):
    """
    This function gets the output path for masked video
    """
    path = f'../Output/{q_no}_{subq_no}'
    if not os.path.exists(path):
        os.makedirs(path)

    path = path + f'/{video_no}_masking.avi'
    return path

def feature_transform(video_frames, q_no, subq_no, video_no):
    """
    This function is Q1 A part where we need to find flow and stabilization using features
    """

    transforms = get_frame_transformation(video_frames)
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory, 50)

    difference = smoothed_trajectory - trajectory
    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # Uncomment this to visualize plots for smoothing
    # plot_trajectory_smoothing_diff(trajectory, smoothed_trajectory)
    path = get_video_path(q_no, subq_no, video_no)
    output_video = VideoWriter(path)

    for i in range(0, len(video_frames)-2):
        frame = video_frames[i]
        w, h = frame.shape[1], frame.shape[0]

        m = get_transformation_matrix_from_affine_values(transforms_smooth[i])        
        adjusted_frame = cv2.warpPerspective(frame, m, (w,h))
        frame_out = cv2.hconcat([frame, adjusted_frame])
        output_video.writeFrame(frame_out)
        # cv2.imshow("Before and After", frame_out)
        # cv2.waitKey(10)

def feature_transform_stable(video_frames, q_no, subq_no, video_no):
    """
    This function is Q2 B part where we need to find flow and stabilization using stable features
    """

    transforms = get_frame_transformation_stable(video_frames)
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory, 50)

    difference = smoothed_trajectory - trajectory
    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # Uncomment this to visualize plots for smoothing
    # plot_trajectory_smoothing_diff(trajectory, smoothed_trajectory)
    path = get_video_path(q_no, subq_no, video_no)
    output_video = VideoWriter(path)

    for i in range(0, len(video_frames)-2):
        frame = video_frames[i]
        w, h = frame.shape[1], frame.shape[0]

        m = get_transformation_matrix_from_affine_values(transforms_smooth[i])        
        adjusted_frame = cv2.warpPerspective(frame, m, (w,h))
        frame_out = cv2.hconcat([frame, adjusted_frame])
        output_video.writeFrame(frame_out)
        # cv2.imshow("Before and After", frame_out)
        # cv2.waitKey(10)



def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def get_translation(flow):
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    ang = ang*180/np.pi
    dx = - np.mean(flow[:,:,0].T)
    dy = - np.mean(flow[:,:, 1].T)
    da = 0
    # Reference: https://stackoverflow.com/questions/60743485/how-to-quantify-difference-between-frames-using-optical-flow-estimation
    return (dx, dy, da)

def get_flow_transform(flow):
    translation = get_translation(flow)
    # mat = np.eye(3,3)
    # mat[0, 2] = -dx
    # mat[1, 2] = -dy
    mat = get_transformation_matrix_from_affine_values(translation)
    return mat
    
def make_mask(flow1, flow2, frame):
    """
    This function makes the mask of the flow and frame simultaneously
    """
    flow_mean1 = np.mean(flow1, axis = (0,1))
    flow_mean2 = np.mean(flow2, axis = (0,1))
    bin_image = np.random.randint(0,255,(flow2.shape[0],flow2.shape[1]))
    for x in range(flow1.shape[0]):
        for y in range(flow1.shape[1]):
            l2_norm1 = np.sqrt(((flow1[x][y][0] - flow_mean1[0])**2) + ((flow1[x][y][1] - flow_mean1[1])**2))
            l2_norm2 = np.sqrt(((flow2[x][y][0] - flow_mean2[0])**2) + ((flow2[x][y][1] - flow_mean2[1])**2))
            if l2_norm1 < 5:
                bin_image[x][y] = 255
            else:
                bin_image[x][y] = 0

            if l2_norm2 < 5:
                frame[x][y] = 255
            else:
                frame[x][y] = 0
    return bin_image, frame
    
def get_masked_flow(flow, mask):
    """
    This function gets the masked flow 
    """
    for x in range(flow.shape[0]):
        for y in range(flow.shape[1]):
            if mask[x][y] == 0:
                flow[x][y][0] = 0
                flow[x][y][1] = 0
    return flow

def get_masked_video(framelist, path):
    """
    This function writes the masked video 
    """
    output_video = VideoWriter(path)
    for each in framelist:
        output_video.writeFrame(each)
        
        
def optical_transform(video_frames, q_no, subq_no, video_no):
    """
    This function uses optical flow method without smoothing 
    """
    prev_frame = video_frames[0]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    
    path = get_video_path(q_no, subq_no, video_no)
    output_video = VideoWriter(path)
    
    for i in range(1, len(video_frames)-1):
        curr_frame = video_frames[i]
        w, h = curr_frame.shape[1], curr_frame.shape[0]
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        mat = get_flow_transform(flow)
        adjusted_frame = cv2.warpPerspective(curr_frame, mat, (w,h))
        frame_out = cv2.hconcat([curr_frame, adjusted_frame])
        output_video.writeFrame(frame_out)
        cv2.imshow("Before and After", frame_out)
        cv2.waitKey(10)
        # prev_frame = adjusted_frame
        prev_frame = curr_frame
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    


def optical_transform_masked(video_frames, q_no, subq_no, video_no):
    """
    This function uses optical flow method without smoothing but with masking
    """
    prev_frame = video_frames[0]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    
    path = get_video_path(q_no, subq_no, video_no)
    output_video = VideoWriter(path)
    
    for i in range(1, len(video_frames)-1):
        curr_frame = video_frames[i]
        w, h = curr_frame.shape[1], curr_frame.shape[0]
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        mask, frame = make_mask(flow, video_1.getFlow(i), curr_frame)
        masked_flow = get_masked_flow(flow, mask)
        mat = get_flow_transform(masked_flow)
        adjusted_frame = cv2.warpPerspective(curr_frame, mat, (w,h))
        frame_out = cv2.hconcat([curr_frame, adjusted_frame])
        output_video.writeFrame(frame_out)
        cv2.imshow("Before and After", frame_out)
        cv2.waitKey(10)
        # prev_frame = adjusted_frame
        prev_frame = curr_frame
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        
def optical_transform_with_smoothing_masked(video_frames, q_no, subq_no, video_no):
    """
    This function uses optical flow method with smoothing and masking 
    """
    prev_frame = video_frames[0]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    transforms = []
    mask_frames = []
    
    for i in range(1, len(video_frames)-1):
        curr_frame = video_frames[i]
        w, h = curr_frame.shape[1], curr_frame.shape[0]
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        mask, frame = make_mask(flow, video_1.getFlow(i), curr_frame)
        mask_frames.append(frame)
        masked_flow = get_masked_flow(flow, mask)
        mat = get_flow_transform(masked_flow)
        trans = get_translation(masked_flow)
        transforms.append(trans)
        adjusted_frame = cv2.warpPerspective(curr_frame, mat, (w,h))
        prev_frame = adjusted_frame
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    path = get_video_path(q_no, subq_no, video_no)
    output_video = VideoWriter(path)

    # uncomment this function and run the program to see the masked video 
    # get_masked_video(mask_frames, path)

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory, 30)
   
    difference =  trajectory - smoothed_trajectory 
    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    plot_trajectory_smoothing_diff(trajectory, transforms_smooth)

   

    for i in range(1, len(video_frames)-2):
        frame1 = video_frames[i]
        w, h = frame1.shape[1], frame1.shape[0]
        m = get_transformation_matrix_from_affine_values(transforms_smooth[i-1])        
        adjusted_frame1 = cv2.warpPerspective(frame1, m, (w,h))
        # frame_out = adjusted_frame[40:320, 40:600]
        frame_out = cv2.hconcat([frame1, adjusted_frame1])
        output_video.writeFrame(frame_out)

def optical_transform_with_smoothing(video_frames, q_no, subq_no, video_no):
    """
    This function uses optical flow method with smoothing but no masking 
    """
    prev_frame = video_frames[0]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    
    path = get_video_path(q_no, subq_no, video_no)
    output_video = VideoWriter(path)
    transforms = []

    
    for i in range(1, len(video_frames)-1):
        curr_frame = video_frames[i]
        w, h = curr_frame.shape[1], curr_frame.shape[0]
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        mat = get_flow_transform(flow)
        trans = get_translation(flow)
        transforms.append(trans)
        adjusted_frame = cv2.warpPerspective(curr_frame, mat, (w,h))
        prev_frame = adjusted_frame
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory, 30)

    difference =  trajectory - smoothed_trajectory 
    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # plot_trajectory_smoothing_diff(trajectory, transforms_smooth)

    path = get_video_path(q_no, subq_no, video_no)
    output_video = VideoWriter(path)

    for i in range(1, len(video_frames)-2):
        frame = video_frames[i]
        w, h = frame.shape[1], frame.shape[0]
        m = get_transformation_matrix_from_affine_values(transforms_smooth[i-1])        
        adjusted_frame = cv2.warpPerspective(frame, m, (w,h))
        # frame_out = adjusted_frame[40:320, 40:600]
        frame_out = cv2.hconcat([frame, adjusted_frame])
        output_video.writeFrame(frame_out)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--question',  help='Question number to decide type of stabilization', default=1, type=int)
    parser.add_argument('--subQuestion',  type=str , default = 'A', help='Decide on type of motion estimation')
    parser.add_argument('--video', default=1, type=int, help='Video to process')

    args = parser.parse_args()

    video_1 = VideoReader("../data",args.question , args.video)
    all_frames = video_1.getFrames()
    if args.question == 1:
        if args.subQuestion == 'A':
            start = time.process_time()
            feature_transform(all_frames, args.question, args.subQuestion, args.video)
            time_taken = time.process_time() - start
            time_per_frame = time_taken/len(all_frames)
            print("Time per frame: ", time_per_frame)
        else:
            start = time.process_time()
            optical_transform_with_smoothing(all_frames, args.question, args.subQuestion, args.video)
            # optical_transform(all_frames, args.question, args.subQuestion, args.video)
            time_taken = time.process_time() - start
            time_per_frame = time_taken/len(all_frames)
            print("Time per frame: ", time_per_frame)
    else:
        if args.subQuestion == 'B':
            start = time.process_time()
            feature_transform_stable(all_frames, args.question, args.subQuestion, args.video)
            time_taken = time.process_time() - start
            time_per_frame = time_taken/len(all_frames)
            print("Time per frame: ", time_per_frame)
        else:
            start = time.process_time()
            optical_transform_with_smoothing_masked(all_frames, args.question, args.subQuestion, args.video)
            # optical_transform(all_frames, args.question, args.subQuestion, args.video)
            time_taken = time.process_time() - start
            time_per_frame = time_taken/len(all_frames)
            print("Time per frame: ", time_per_frame)