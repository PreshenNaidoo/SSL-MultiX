# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:30:41 2022

@author: Preshen Naidoo
"""

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import directed_hausdorff
import cv2
import math
import scipy
from drawhelper import *


def euclidean_distance(a,b):
    """
    Calculate the Euclidean distance between two points.

    Args:
        a (tuple or list): Coordinates of the first point (x, y).
        b (tuple or list): Coordinates of the second point (x, y).

    Returns:
        float: The Euclidean distance between points a and b.
    """
    return math.dist(a, b)

def triangle_area(x1, y1, x2, y2, x3, y3):
    """
    Calculate the area of a triangle given its three vertices.

    Args:
        x1, y1 (float): Coordinates of the first vertex.
        x2, y2 (float): Coordinates of the second vertex.
        x3, y3 (float): Coordinates of the third vertex.

    Returns:
        float: The area of the triangle.
    """
    return abs(0.5 * (((x2-x1)*(y3-y1))-((x3-x1)*(y2-y1))))

def get_polyline_area(curve_points):
    """
    Calculate the area enclosed by a polyline.

    Args:
        curve_points (numpy.ndarray): Array of (x, y) points representing the polyline.

    Returns:
        float: The area enclosed by the polyline. Returns 0 if the polyline has fewer than 2 points.
    """
    if(len(curve_points) <=1):
        return 0
    
    return cv2.contourArea(curve_points)

def get_angles(pt1, pt2, pt3):
    """
    Calculate the internal angles of a triangle formed by three points.

    Args:
        pt1, pt2, pt3 (numpy.ndarray): Coordinates of the three points.

    Returns:
        tuple: Three angles (in degrees) of the triangle formed by pt1, pt2, and pt3.
    """
    A,B,C = pt1, pt2, pt3

    e1 = B-A; e2 = C-A
    denom = np.linalg.norm(e1) * np.linalg.norm(e2)
    d1 = np.rad2deg(np.arccos(np.dot(e1, e2)/denom))
    
    e1 = C-B; e2 = A-B
    denom = np.linalg.norm(e1) * np.linalg.norm(e2)
    d2 = np.rad2deg(np.arccos(np.dot(e1, e2)/denom))
    
    d3 = 180-d1-d2
    
    return d1, d2, d3


def unit_vector(vector):
    """
    Return the unit vector of a given vector.

    Args:
        vector (numpy.ndarray): Input vector.

    Returns:
        numpy.ndarray: Unit vector in the same direction as the input vector.
    """
    return vector / np.linalg.norm(vector)

def get_angle_between_vectors(v1, v2):
    
    """
    Calculate the angle (in radians) between two vectors.

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793

    Returns:
        float: The angle in radians between the two vectors.
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    
def compute_tortuosity(curve_points):
    """
    Compute the tortuosity of a curve (a measure of roughness or smoothness).

    Args:
        curve_points (numpy.ndarray): Array of (x, y) points representing the curve.

    Returns:
        tuple: Tortuosity value (between 0 and 1) and the number of sharp turns.
    """
    
    #Approximate before iterating to reduce the number of unnecessary points
    #since we don't want an edge defined by adjacent pixel points.
    new_points = cv2.approxPolyDP(curve_points, epsilon = 1e-3, closed = False)
    #The input array, but with all or a subset of the dimensions of length 1 removed.
    new_points = np.squeeze(new_points)
    
    if(len(new_points) < 4):
        return 0,0  #very high roughness
    
    start_pt = new_points[0]
    end_pt = new_points[-1]
    
    if(start_pt[0] == end_pt[0] and start_pt[1] == end_pt[1]):
        new_points = new_points[:-1]
    
    sum_ang = 0
    cnt = 0
    cnt_sharp = 0 #number of sharp turns
    for i in range(len(new_points)-2):
        pt1 = new_points[i]
        pt2 = new_points[i+1]
        pt3 = new_points[i+2]
        
        v1 = (pt1[0] - pt2[0], pt1[1] - pt2[1])
        v2 = (pt3[0] - pt2[0], pt3[1] - pt2[1])
        
        ang = abs(get_angle_between_vectors(v1,v2))
        
        if(ang <= (math.pi/3)):
            cnt_sharp+=1
        
        sum_ang += ang #abs(get_angle_between_vectors(v1,v2))
        cnt+=1
        
    unit_tort = sum_ang/(cnt * math.pi)
    
    #add penalisations    
    unit_tort -= 0.05 * (cnt_sharp)  #penalize by half a point for every sharp corner(0.1 is a point since range is [0,1])
    
    #Note: penalisations from here are specific to the general shape if a LV curve since
    #we expect the bottom corners of the left ventricle to more than a certain angle.
    #If you want a more generalised function, use only the top half of the functions,
    #i.e. before this 'Note'.
    #Check corner by start point
    pt1 = new_points[3]
    pt2 = new_points[0]
    pt3 = new_points[-1]
    v1 = (pt1[0] - pt2[0], pt1[1] - pt2[1])
    v2 = (pt3[0] - pt2[0], pt3[1] - pt2[1])
    ang = abs(get_angle_between_vectors(v1,v2))
    if(ang <= (math.pi/2.5)):
        unit_tort -= 0.15    #penalize heavily
        cnt_sharp+=1
        
    #Check corner by end point
    pt1 = new_points[-4]
    pt2 = new_points[-1]
    pt3 = new_points[0]
    v1 = (pt1[0] - pt2[0], pt1[1] - pt2[1])
    v2 = (pt3[0] - pt2[0], pt3[1] - pt2[1])
    ang = abs(get_angle_between_vectors(v1,v2))
    if(ang <= (math.pi/2.5)):
        unit_tort -= 0.15    #penalize heavily
        cnt_sharp+=1
        
    if(unit_tort < 0): #if after penalising we have a negative number, make zero
        unit_tort = 0
        
    return unit_tort, cnt_sharp


        
def compute_similarity_score(similarity_curves, curve):
    """
        Compute the similarity score between a given curve and multiple reference curves.

        Args:
            similarity_curves (list): List of reference curves.
            curve (numpy.ndarray): The input curve to be compared.

        Returns:
            float: The top 3 average similarity score (lower values indicate more similarity).
    """
    
    total = 0
    min1 = 9999999.0
    all_sim = []
    
    for sim_curve in similarity_curves:
        #Approximate before iterating to reduce the number of unnecessary points
        #since we don't want an edge defined by adjacent pixel points.
        poly1 = cv2.approxPolyDP(sim_curve, epsilon = 1e-2, closed = False)
        poly2 = cv2.approxPolyDP(curve, epsilon = 1e-2, closed = False)
        poly1 = np.squeeze(poly1)
        poly2 = np.squeeze(poly2)
        # start_pt = poly1[0]
        # end_pt = poly1[-1]
        
        # if(start_pt[0] != end_pt[0] and start_pt[1] != end_pt[1]):
        poly1 = np.append(poly1, [poly1[0]], axis=0)
        poly2 = np.append(poly2, [poly2[0]], axis=0)
        
        sim = cv2.matchShapes(sim_curve, curve, 1, 0.0)
        
        all_sim.append(sim)
        total += sim
        if(sim<min1): 
            min1 = sim
    
    num_sim_curves = len(similarity_curves)
    total = 1
    if (num_sim_curves > 0):
        total/=len(similarity_curves)
    
    if(total > 1): #limit it to 1 max, which wil mean unsimilar
        total = 1
        
    if(min1 > 1): #limit it to 1 max, which wil mean unsimilar
        min1 = 1
        
    all_sim.sort()
    top3 = 1
    if(len(all_sim) > 0):
        top3 = all_sim[0]+all_sim[1]+all_sim[2]
        top3/=3
    
    if(top3 > 1): #limit it to 1 max, which wil mean unsimilar
        top3 = 1
    
    #return total
    #return min1
    return top3

def compute_MAE_from_smooth_curve(y_values, window_length, poly_order = 3):
    """
    Compute the Mean Absolute Error (MAE) from a smoothed version of the input curve.

    Args:
        y_values (numpy.ndarray): Input y-values of the curve.
        window_length (int): Length of the smoothing window.
        poly_order (int): Order of the polynomial for the Savitzky-Golay filter.

    Returns:
        tuple: Mean Absolute Error and the smoothed curve values.
    """
    if(window_length < poly_order):
        return None, None
    
    #also window_length must be less than or equal to the size of x
    
    #Note the rule for savgol filter is that the minimum window length
    #should be higher than the polynomial order. If it is the same, then
    #there is no smoothing. Window length should also be an odd number.
    y_filtered = scipy.signal.savgol_filter(y_values, window_length=window_length, polyorder=3) 
    
    mae = np.mean(np.abs(y_values-y_filtered))
    
    return mae, y_filtered

def get_dice_and_hd(label_curve_points, predicted_curve_points, image_size):
    """
    Compute the Dice coefficient and Hausdorff distance between two sets of curve points.

    Args:
        label_curve_points (numpy.ndarray): Ground truth curve points.
        predicted_curve_points (numpy.ndarray): Predicted curve points.
        image_size (tuple): Size of the image (height, width).

    Returns:
        tuple: Dice coefficient and Hausdorff distance.
    """
    dice = 0
    hd = 10
    
    if(len(label_curve_points) == 0 or len(predicted_curve_points) == 0):
        return 0, 10
    
    zeros = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    img1 = fill_poly_on_image(zeros.copy(), label_curve_points, 1)
    
    img2 = zeros.copy()
    if(len(predicted_curve_points)!=0):
        img2 = fill_poly_on_image(zeros.copy(), predicted_curve_points, 1)
    
    dice = compute_Dice_coefficient(img1, img2)
    hd = compute_Hausdorff_distance(img1, img2)
    
    return dice, hd

def get_dice_and_hd_from_masks(label_mask, predicted_mask):
    """
    Compute the Dice coefficient and Hausdorff distance between two binary masks.

    Args:
        label_mask (numpy.ndarray): Ground truth mask.
        predicted_mask (numpy.ndarray): Predicted mask.

    Returns:
        tuple: Dice coefficient and Hausdorff distance.
    """
    dice = 0
    hd = 0    
    
    dice = compute_Dice_coefficient(label_mask, predicted_mask)
    hd = compute_Hausdorff_distance(label_mask, predicted_mask)
    
    return dice, hd


def compute_Dice_coefficient(original_mask, predicted_mask):
    '''
            2 x Intersection
    DICE = -------------------
           Union + Intersection
           
    This could be seen as:
    
                2 x TP
    DICE = -----------------
           (FP + TP + FN) + TP

    Parameters
    ----------
    original_mask : 2d numpy array
        A binary mask
    predicted_mask : 2d numpy array
        A binary mask

    Returns
    -------
    dc : float
        DIce coefficient value in the range [0,1]

    '''
    
    #a = original_mask.ravel()
    #b = predicted_mask.ravel()
    
    #a = original_mask.flatten()
    #b = predicted_mask.flatten()    
    
    #count1 = (a == 1).sum()
    #count2 = (b == 1).sum()
    
    #count1 = np.count_nonzero(original_mask)
    #count2 = np.count_nonzero(predicted_mask)
    
    a = np.array(original_mask, dtype=np.bool_)
    a = np.atleast_1d(a)
    a = tf.reshape(a, [-1])
    
    b = np.array(predicted_mask, dtype=np.bool_)
    b = np.atleast_1d(b)
    b = tf.reshape(b, [-1])    
    
    #dice original code, first attempt
    #intersection = np.count_nonzero(a & b)    
    #union = np.count_nonzero(a) + np.count_nonzero(b)
    #dc = (2. * intersection)/float(union) # + intersection - intersection
    
    #dice computation, second attempt using scipy function
    #dc = dice(a, b)
    
    #dice computation, third attempt
    dc = np.sum(predicted_mask[original_mask==1])*2.0 / (np.sum(predicted_mask) + np.sum(original_mask))
    
    #All three above produce the same output. The second attemp needs to be subtracted from 1.
    
    return dc


def compute_IoU(original_mask, predicted_mask):    
    '''
            Intersection
    IoU = -------------------
                Union
           
    This could be seen as:
    
                 TP
    IoU = -----------------
           (FP + TP + FN)

    Parameters
    ----------
    original_mask : 2d numpy array
        A binary mask.
    predicted_mask : 2d numpy array
        A binary mask.

    Returns
    -------
    iou : float
        IoU value.

    '''
    
    a = np.array(original_mask, dtype=np.bool)
    a = np.atleast_1d(a)
    a = tf.reshape(a, [-1])
    
    b = np.array(predicted_mask, dtype=np.bool)
    b = np.atleast_1d(b)
    b = tf.reshape(b, [-1])   
    
    #IoU Attempt 1
    #intersection = np.count_nonzero(a & b)    
    #union = (np.count_nonzero(a) + np.count_nonzero(b)) - intersection      
    #iou = intersection/union
    
    #IoU attempt 2
    intersection = np.sum(predicted_mask[original_mask==1])
    iou = (intersection) / ((np.sum(predicted_mask) + np.sum(original_mask)) - intersection)
    
    #Attempt 1 and 2 above produce the same output. i.e. both works.
    
    return iou



def compute_Hausdorff_distance(original_mask, predicted_mask):
    '''
    We're going to use the Distance Map(Distance Transform) method of computing the Hausdorf distance.
    A python library Scipy has a function that can do this for us called distance_transform_edt.
    After computing the distance map for mask2 i.e DM2, we need to overlap the boundary of mask1 onto 
    DM2. The take the maximum value in DM2 where mask1 overlaps. 
    This maximum value will be the Hausdorf distance d(mask1, mask2).
    Then we need to find the Hausdorf distance d(mask2, mask1).
    The final distance will be the max(d(mask1, mask2), d(mask2, mask1))
    
    However, first lets try to do this using scipy.spatial.distance.directed_hausdorf
    
    Note: for distance, should we use Euclidean or Manhattan etc?
    
    https://en.wikipedia.org/wiki/Distance_transform
    
    https://cs.stackexchange.com/questions/117989/hausdorff-distance-between-two-binary-images-according-to-distance-maps
    
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.morphology.distance_transform_edt.html
    
    Paper: AN IMAGE ALGORITHM FOR COMPUTING THE SDORFF DISTANCE EFFICIENTLY IN LINEAR TIME
    Paper: A Linear Time Algorithm of Computing Hausdorff Distance for Content-based Image Analysis    
    
    '''
    
    hd1 = directed_hausdorff(original_mask, predicted_mask)[0]
    hd2 = directed_hausdorff(predicted_mask, original_mask)[0]
    
    return max(hd1, hd2)    
    #return 0