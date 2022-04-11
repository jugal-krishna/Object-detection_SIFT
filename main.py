import cv2
import numpy as np
import matplotlib.pyplot as plt

# Loading the images

target_1 = cv2.imread('./HW3_Data/dst_0.jpg')
target_2 = cv2.imread('./HW3_Data/dst_1.jpg')
obj_1 = cv2.imread('./HW3_Data/src_0.jpg')
obj_2 = cv2.imread('./HW3_Data/src_1.jpg')
obj_3 = cv2.imread('./HW3_Data/src_2.jpg')
targets = [target_1, target_2]
objects = [obj_1, obj_2, obj_3]
images = [target_1, target_2, obj_1, obj_2, obj_3]


# SIFT

def SIFT(img):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


# Function to apply SIFT, BF Matcher , RANSAC and Homography

def img_match(img1, img2):
    # SIFT

    kp1, des1 = SIFT(img1)
    kp2, des2 = SIFT(img2)

    # Brute force Matcher

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test for good matches

    good = []  # Good Matches
    for m, n in matches:
        if m.distance < 0.8 * n.distance:   # Acc to the paper: D.Lowe, "Distinctive image features from scale-invariant keypoints," International Journal of Computer Vision,2004
            good.append(m)
    img_matched = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    good.sort(key=lambda x: x.distance)
    # Top-20 scoring matches found by the matcher before the RANSAC operation
    img_matched_top_20 = cv2.drawMatches(img1, kp1, img2, kp2, good[:20], None,
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Homography and RANSAC

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)  # M-Homography matrix,
        matchesMask = mask.ravel().tolist()
        dst_projected = cv2.perspectiveTransform(src_pts, M)
        # img2 = cv2.polylines(img2, np.int32(dst_projected), True, 255, 3, cv2.LINE_AA)
        draw_params = dict(matchColor=(0, 255, 0),  # Green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # Inliners
                           flags=2)
        img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

        # Top 10 matches

        errors = []  # Saving errors in the list for sorting matches wrt them
        for i in range(len(dst_projected)):
            errors.append([i, ((dst_projected[i][0][0] - dst_pts[i][0][0]) ** 2 +
                               (dst_projected[i][0][1] - dst_pts[i][0][1]) ** 2)
                           ** (0.5)])  # Appending errors (Eucledian Distance) along with index for sorting
            errors.sort(key=lambda a: a[1])    # Sorting errors

        top_10 = np.zeros((dst_projected.shape[0], 1)) # Initializing Mask values for top-10 inliners
        for i in range(10):
            top_10[int(errors[i][0])][0] = 1   # Set mask values to 1 for top-10 inliners

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           matchesMask=top_10,  # Top 10 inliners
                           flags=cv2.DrawMatchesFlags_DEFAULT)
        img_ransac_top_10 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    return img_matched, img_matched_top_20, img_ransac, mask.sum(), img_ransac_top_10, M

# Outputs

# Saving images before and after RANSAC matching

j = 1
for target in targets:
    for object in objects:
        img_matched, img_matched_top_20, img_ransac, sum, img_ransac_top_10, M = img_match(object, target)
        print(f"Total number of inliners in {j}: {sum}")
        print(f"Homography matrix {j} :\n {M} \n")
        cv2.imwrite(f'HW3_Data/Img_sols/Matches/im_matched_{j}.png', img_matched)
        cv2.imwrite(f'HW3_Data/Img_sols/Matches/im_matched_top20_{j}.png', img_matched_top_20)
        cv2.imwrite(f'HW3_Data/Img_sols/Matches/im_ransac_{j}.png', img_ransac)
        cv2.imwrite(f'HW3_Data/Img_sols/Matches/im_ransac_top10_{j}.png', img_ransac_top_10)
        j += 1

#Saving images with keypoints

i = 1
for image in images:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to gray for better visibility of features
    kp, des = SIFT(image)
    print(f"No. of keypoints in img{i} = {len(kp)}")
    img = cv2.drawKeypoints(image, kp, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #img = cv2.drawKeypoints(gray, kp, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(f'HW3_Data/Img_sols/keypoints/image_{i}.png', img)
    i += 1