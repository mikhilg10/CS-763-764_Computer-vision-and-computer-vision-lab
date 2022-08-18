import cv2
import os,sys
stitcher = cv2.createStitcher(False)

imgDirPath = sys.argv[1]
imageNames = []
for file in os.listdir(imgDirPath):
    if os.path.isfile(os.path.join(imgDirPath, file)):    
        imageNames.append(os.path.join(imgDirPath, file))

if len(imageNames)!=2:
    raise ValueError("Directory should have exactly 2 images")

print(imageNames)
img1 = cv2.imread(imageNames[0])
img2 = cv2.imread(imageNames[1])

result = stitcher.stitch((img1,img2))

cv2.imshow("Stitcher API Image", result[1])
cv2.waitKey(0)
cv2.imwrite("../results/Stitcher_API_2_images_auto_mosaic_campus.jpg", result[1])


'''
tried for stitching multiple images one-by-one using Stitcher API
uncomment to run for 5 images
'''
# img0 = cv2.imread(imageNames[0])
# img1 = cv2.imread(imageNames[1])
# img2 = cv2.imread(imageNames[2])
# img3 = cv2.imread(imageNames[3])
# img4 = cv2.imread(imageNames[4])

# result = stitcher.stitch((img1,img2))
# cv2.imshow("Stitcher API Image", result[1])
# cv2.waitKey(0)

# result = stitcher.stitch((result[1],img3))
# cv2.imshow("Stitcher API Image", result[1])
# cv2.waitKey(0)

# result = stitcher.stitch((result[1],img0))
# cv2.imshow("Stitcher API Image", result[1])
# cv2.waitKey(0)