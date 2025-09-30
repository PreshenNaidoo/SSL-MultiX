# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:05:00 2022

@author: Preshen Naidoo
"""

import cv2
import numpy as np

def get_binary_image(mask):
    """
    Converts a 3-channel mask image to a binary (0 or 1) image.

    Parameters:
        mask (numpy.ndarray): Input image with shape (height, width, 3).

    Returns:
        numpy.ndarray: Binary image with values 0 and 1, shape (height, width).
    """
    img_temp_np = np.zeros((mask.shape[0],mask.shape[1]))
    for i in range(mask.shape[0]):
         for j in range(mask.shape[1]):
            if mask[i,j][0] > 0:            
                img_temp_np[i,j] = 1
            else:
                img_temp_np[i,j] = 0

    img = img_temp_np.astype(np.uint8)
    
    return img

def get_rgb_image(predicted_mask, colour = (int(255),int(255),int(255))):
    """
    Converts a binary mask image to an RGB image with a specified color for mask regions.

    Parameters:
        predicted_mask (numpy.ndarray): Binary mask with values 0 and 1.
        colour (tuple): RGB color for mask regions, default is white (255, 255, 255).

    Returns:
        numpy.ndarray: RGB image with mask regions in the specified color.
    """

    img_temp_np = np.zeros((predicted_mask.shape[0],predicted_mask.shape[1],3))

    for i in range(predicted_mask.shape[0]):
         for j in range(predicted_mask.shape[1]):
             if predicted_mask[i,j] == 1:
                img_temp_np[i,j] = colour
             else:
                img_temp_np[i,j] = (int(0),int(0),int(0))

    img = img_temp_np.astype(np.uint8)
    
    return img

def get_overlay_image(echo_image, gt, pred):
    """
    Overlays contours of ground truth and predicted masks on an image.

    Parameters:
        echo_image (numpy.ndarray): Base image on which contours are drawn.
        gt (numpy.ndarray): Ground truth binary mask.
        pred (numpy.ndarray): Predicted binary mask.

    Returns:
        tuple: Overlay image with drawn contours and the largest contour points from prediction.
    """

    #Get polyline of the mask    
    image_test1_gray = gt#cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(image_test1_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_index_area = 0
    largest_area = -1
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if (area > largest_area):
            largest_area = area
            largest_index_area = i

    #Get the shape/polyline of the largest area
    img8 = echo_image
    if largest_area != -1:
        poly_points = contours[largest_index_area]
        img8 = draw_poly_on_image_weight(echo_image, poly_points, (0, 0, 255), 2)

    #Get polyline of the mask    
    image_test1_gray = pred#cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(image_test1_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_index_area = 0
    largest_area = -1
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if (area > largest_area):
            largest_area = area
            largest_index_area = i

    #Get the shape/polyline of the largest area
    img8p = img8
    poly_pointsp = None
    if largest_area != -1:
        poly_pointsp = contours[largest_index_area]
        img8p = draw_poly_on_image_weight(img8, poly_pointsp, (0, 255, 0), 2)


    return img8p, poly_pointsp

def get_overlay_image_pts(echo_image, gt_image_file, pred_points):
    """
    Draws ground truth contours and predicted points as overlays on an image.

    Parameters:
        echo_image (numpy.ndarray): Base image on which overlays are drawn.
        gt_image_file (numpy.ndarray): Ground truth binary mask.
        pred_points (numpy.ndarray): Predicted contour points.

    Returns:
        numpy.ndarray: Image with ground truth and predicted overlays.
    """

    #Get polyline of the mask
    image_test1_gray = gt_image_file#cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(image_test1_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_index_area = 0
    largest_area = -1
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if (area > largest_area):
            largest_area = area
            largest_index_area = i

    #Get the shape/polyline of the largest area
    img8 = echo_image
    if largest_area != -1:
        poly_points = contours[largest_index_area]
        img8 = draw_poly_on_image_weight(echo_image, poly_points, (0, 0, 255), 2)

    #Get polyline of the mask
    poly_pointsp = pred_points
    img8p = draw_poly_on_image_weight(img8, poly_pointsp, (0, 255, 0), 2)


    return img8p

def draw_poly_on_image(img, coordinates, color, use_weight=True):
    """
    Draws a polygon on an image.

    Parameters:
       img (numpy.ndarray): Input image.
       coordinates (list): List of polygon coordinates.
       color (tuple): RGB color of the polygon.
       use_weight (bool): Whether to use weighted thickness for drawing.

    Returns:
       numpy.ndarray: Image with drawn polygon.
    """

    pts = np.array(coordinates, np.int32)
    pts = pts.reshape((-1,1,2))
    
    weight = 1
    if (use_weight):
        weight = 2
        
    img = cv2.polylines(img, [pts], True, color, thickness=weight)
    return img


def draw_poly_on_image_weight(img, coordinates, color, weight=1):
    """
    Draws a polygon on an image with a specified line thickness.

    Parameters:
        img (numpy.ndarray): Input image.
        coordinates (list): List of polygon coordinates.
        color (tuple): RGB color of the polygon.
        weight (int): Thickness of the polygon lines.

    Returns:
        numpy.ndarray: Image with drawn polygon.
    """

    pts = np.array(coordinates, np.int32)
    pts = pts.reshape((-1,1,2))    
        
    img = cv2.polylines(img, [pts], True, color, thickness=weight)
    return img


def fill_poly_on_image(img, coordinates, color):
    """
    Fills a polygon on an image with a specified color.

    Parameters:
        img (numpy.ndarray): Input image.
        coordinates (list): List of polygon coordinates.
        color (tuple): RGB color to fill the polygon.

    Returns:
        numpy.ndarray: Image with filled polygon.
    """

    pts = np.array(coordinates, np.int32)
    pts = pts.reshape((-1,1,2))
    img = cv2.fillPoly(img, [pts], color)
    return img


def draw_line_on_image(img, x1, y1, x2, y2, color, use_weight=True):
    """
    Draws a line on an image between two points.

    Parameters:
        img (numpy.ndarray): Input image.
        x1, y1 (int): Starting point of the line.
        x2, y2 (int): Ending point of the line.
        color (tuple): RGB color of the line.
        use_weight (bool): Whether to use weighted thickness for the line.

    Returns:
        numpy.ndarray: Image with drawn line.
    """

    weight = 1
    if (use_weight):
        weight = 2
        
    img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=weight)
    return img

