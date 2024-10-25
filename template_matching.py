#######################################################################
#   
#   File:           template_matching.py
#   Project:        HAARTA Analyzer
#   Date:           20-08-2021
#   Author:         Danjel Keekstra
#   Discription:    Template matching with sub-pixel precision
#
#######################################################################
# imports
#######################################################################
import numpy as np
import cv2
from scipy import interpolate
from scipy import optimize

#######################################################################
# Helpers
#######################################################################
def __pull_matrix(size, frame, loc):
    size = int((size -1) /2)
    mat = []
    for i in range(loc[1] - size, loc[1] + size +1):
        mat.append(frame[i][loc[0] - size: loc[0] + size + 1])
    return mat

def __find_max_subpixel_roi(frame, templ, pos, offset = 50, matsize = 11):
    h, w = templ.shape
    fh, fw = frame.shape
    xs = np.around(pos[0] - w/2 - offset).astype(int)
    xs = 0 if xs < 0 else xs
    xe = np.around(pos[0] + w/2 + offset).astype(int)
    xe = fw -1 if xe >= fw else xe
    ys = np.around(pos[1] - h/2 - offset).astype(int)
    ys = 0 if ys < 0 else ys
    ye = np.around(pos[1] + h/2 + offset).astype(int)
    ye = fh -1 if ye >= fh else ye
   
    roi = frame[ys:ye, xs:xe]
    res = cv2.matchTemplate(roi, templ, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    mat = np.array(__pull_matrix(matsize, res, max_loc))

    xy = np.arange(0, matsize, 1)     
    f = interpolate.interp2d(xy, xy, mat, kind='cubic')

    def __nf(d): 
        return -f(d[0],d[1])
    max_loc2 = optimize.fmin(__nf, [(matsize-1)/2,(matsize-1)/2], disp=False)

    (x, y) = templ.shape
    nx = max_loc[0] - (2 - max_loc2[0]) + y/2
    ny = max_loc[1] - (2 - max_loc2[1]) + x/2

    #compensate for the roi
    nx = xs + nx  
    ny = ys + ny
    return [nx, ny]

#######################################################################
# template matching
#######################################################################
def template_tracking(frame_array, template, initial_position, roi_offset_size = 50, interpol_mat_size = 11):
    """Function for tracking a single template in a video with template matching with subpixel precision.

    Args:
        frame_array (ndarray): Array with frames of the video with 'int8' as type.

        template (ndarray): Template for the template matching with sub-pixel precision with 'int8' as type.

        initial_position (array): Location of the template in the first frame ([x, y]) with 'int' as type.

        roi_offset_size (int, optional): Size in pixels of the region around the initial or last position to match the template to. Defaults to 50.

        interpol_mat_size (int, optional): Size in pixels of the peak region to apply interpolation on. Defaults to 11.
    Returns:
        points (array): Array with detected points with 'float' as type. [[x0, y0], ... [xn, yn]] length is the same as the frame length.
    """
    points = []
    currp = initial_position
    for frame in frame_array:
        currp = __find_max_subpixel_roi(frame, template, currp, roi_offset_size, interpol_mat_size) 
        points.append(currp)
    return points    

def template_tracking_dual(frame_array, left_template, right_template, initial_position, roi_offset_size = 50, interpol_mat_size = 11):
    """Function for tracking two templates in a video with template matching with subpixel precision.

    Args:
        frame_array (ndarray): Array with frames of the video with 'uint8' as type.

        left_template (ndarray): Template for the template matching with sub-pixel precision with 'uint8' as type.

        right_template (ndarray): Template for the template matching with sub-pixel precision with 'uint8' as type.

        initial_position (array): Location of the template in the first frame [[left_x, left_y], [right_x, right_y]] with 'int' as type.

        roi_offset_size (int, optional): Size in pixels of the region around the initial or last position to match the template to. Defaults to 50.

        interpol_mat_size (int, optional): Size in pixels of the peak region to apply interpolation on. Defaults to 11.

    Returns:
        points (array): Array with detected points with 'float' as type. [[[xl0, yl0], [xr0, yr0]], ... [[xln, yln], [xrn, yrn]] length is the same as the frame length.
    """
    points = []
    pl = initial_position[0]
    pr = initial_position[1]
    for frame in frame_array:
        pl = __find_max_subpixel_roi(frame, left_template, pl, roi_offset_size, interpol_mat_size) 
        pr = __find_max_subpixel_roi(frame, right_template, pr, roi_offset_size, interpol_mat_size)
        points.append([pl, pr])
    return points   
#######################################################################