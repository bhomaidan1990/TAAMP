#!/usr/bin/env python2
import cv2
import numpy as np

# bgr
blue = (255, 0, 0)
white = (255, 255, 255)
black = (0, 0, 0)
pink = (255, 192, 203)
green = (0, 255, 0)
red = (0, 0, 255)
yellow = (0, 255, 255)
purple = (201, 35, 118)

# detect blob on binary image
def detect_blob(image, threshold_low, threshold_high):
    #lower_blue = np.array([110,50,50])
    #upper_blue = np.array([130,255,255])
    
    color_mask = cv2.inRange(image, threshold_low, threshold_high)
    
    kernel = np.ones((3, 3), np.uint8)
    # remove noises
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    # recover points should be included
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def find_corners(color_image, workspace):
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow("gray", gray_image)

    #cv2.waitKey(1)
    
    # find Harris corners
    gray = np.float32(gray_image)
    #dst = cv2.cornerHarris(gray, 7, 7, 0.04)
    dst = cv2.cornerHarris(gray, 9, 9, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)
    
    ## find centroids
    contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    within_boundary_contours = []
    for contour in contours:
        if check_contour_in_boundary(contour, workspace):
            within_boundary_contours.append(contour)
    
    centroids = []
    for contour in within_boundary_contours:
        centroid = get_centroid(contour)
        if centroid is not None:
            centroids.append([centroid.tolist()])
    
    centroids = np.array(centroids)
    
    corners = np.float32(centroids.copy())
    
    #print("centroids")
    #print(centroids)
    #print("centroids shape: ", centroids.shape)
    ## define the criteria to stop and refine the corners
    if len(corners) != 0:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        #cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), criteria)
        cv2.cornerSubPix(gray_image, corners, (7, 7), (-1, -1), criteria)
    
    return corners

def check_contour_in_boundary(contour, boundary):
    total = 0
    inside = 0
    
    for point in contour:
        if cv2.pointPolygonTest(boundary, tuple(point[0]), False) != -1:
            inside += 1
        total += 1
    
    return inside / total > 0.8

def get_contour_closest_distance(contour_1, contour_2):
    min_distance = float('inf')
    
    for point_1 in contour_1:
        for point_2 in contour_2:
            distance = cv2.norm(point_1, point_2)
            min_distance = min(distance, min_distance)
    #return cv2.norm(get_centroid(contour_1) - get_centroid(contour_2)) only compare centroid for faster result
    
    return min_distance

# it is ok to use convexhull here, but it is a tool with different shape
# another method should be used
# like if they are still close enough, apply cv2.MORPH_CLOSE repetitively
def prune_contours(contours, boundary, threshold, corners, corner_epsilon = 0.06, convex = True):
    within_boundary_contour = []
    
    #prune the contours outside of boundary
    for contour in contours:
        if boundary is None or check_contour_in_boundary(contour, boundary):
            within_boundary_contour.append(contour)

    # merge the contours that are close to each other
    #merged_contours = []
    #for contour in within_boundary_contour:
        #find_group = False
        #for merged_contour_index in range(len(merged_contours)):
            #if get_contour_closest_distance(contour, merged_contours[merged_contour_index]) < threshold:
                #find_group = True
                #merged_contours[merged_contour_index] = np.r_[merged_contours[merged_contour_index], contour]
                #if convex:
                    #merged_contours[merged_contour_index] = cv2.convexHull(np.array(merged_contours[merged_contour_index]))
                #else:
                    #merged_contours[merged_contour_index] = np.array(merged_contours[merged_contour_index])
        #if not find_group:
            #if convex:
                #merged_contours.append(cv2.convexHull(np.array(contour)))
            #else:
                #merged_contours.append(np.array(contour))
    
    # find the largest grouped item as the target
    largest_contour = None
    largest_area = -float('inf')
    
    for contour in within_boundary_contour:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    #if largest_contour is not None:
        #epsilon = 0.1 * cv2.arcLength(largest_contour, True)
        #largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    new_corners_float = []
    is_corners_only = False
    
    if largest_contour is not None:
        largest_contour = cv2.convexHull(largest_contour)
        
        new_corners = largest_contour.tolist()
        for corner in corners:
            if boundary is None or check_contour_in_boundary(contour, boundary):
                epsilon = corner_epsilon * cv2.arcLength(largest_contour, True)
                if cv2.pointPolygonTest(largest_contour, tuple(corner[0]), True) > -epsilon:
                    #print("append", [corner[0][0], corner[0][1]])
                    new_corners_float.append([corner[0][0], corner[0][1]])
                    new_corners.append([[int(round(corner[0][0])), int(round(corner[0][1]))]])
        
        is_corners_only = len(new_corners_float) == 4
        
        
        #print("new_corners_float", new_corners_float)
        #rect = cv2.minAreaRect(largest_contour)
        #largest_contour = cv2.cv.BoxPoints(rect)
        #largest_contour = np.array([largest_contour])
        #largest_contour = np.int0(largest_contour)

    return largest_contour, new_corners_float, is_corners_only

def get_centroid(contour):
    m = cv2.moments(contour)
    if m["m00"] == 0.0:
        return None
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])

# ignore NaN
def get_3d_centroid(x_y_z):
    return np.nanmean(x_y_z, axis = 1)

# currently only works with depth image encoding with zeros turned to nan
# return all the indices and contents within the contour
# it returns the list of indices
# [[0.0, 0.0], [0.0, 1.0],...]
# and z
# [1.0, 1.0, 1.0, ...]
# z unit is in mm, may need an offset parameter later
def get_all_points_in_contour(contour, image):
    mask_image = np.zeros(image.shape)
    # last -1 to fill the shape
    cv2.drawContours(mask_image, [contour], -1, white, -1)
    indices = np.nonzero(mask_image)
    z = image[indices[0], indices[1]]
    
    #print("indices")
    #print(indices)
    
    #print("z")
    #print(z)
    
    # exclude points with large z offset (which implies depth detection error)
    z_mean = np.nanmean(z)
    z_std = np.nanstd(z)
    #print("mean", z_mean)
    #print("std", z_std)
    #same_depth_image = np.where(image < z_mean - z_std or image > z_mean + z_std, 0.0, 1.0)
    convert_nan_image = image.copy()
    convert_nan_image[np.isnan(convert_nan_image)] = 0
    close_depth_image = np.where(convert_nan_image > z_mean - z_std, 1.0, 0.0)
    further_depth_image = np.where(convert_nan_image < z_mean + z_std, 1.0, 0.0)
    same_depth_image = np.logical_and(close_depth_image, further_depth_image)
    mask_image = np.logical_and(mask_image, same_depth_image).astype(float)
    
    indices = np.nonzero(mask_image)
    z = image[indices[0], indices[1]]
    
    return np.array([indices[1], indices[0]]).T * 1.0, np.array(z) * 1.0

def remove_zero_in_depth_image(depth_image):
    removed_zero_image = depth_image.astype('float')
    removed_zero_image[removed_zero_image == 0] = np.nan
    return removed_zero_image