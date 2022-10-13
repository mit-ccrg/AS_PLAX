from pkg_resources import safe_extra
import numpy as np
import cv2
import os
from read_dicom import read_dicom

def de_identify_study(study_dir, save_dir):
    """
    de-identify all videos in a study and save them as npy files in save_dir
    """
    file_list = os.listdir(study_dir)
    count = 0
    for file in file_list:
        if 'txt' in file:
            continue
        try:
            pixels, pixel_dim, n_frames = read_dicom(os.path.join(study_dir, file))
        except:
            continue
        if n_frames > 1:
            video_out = de_identify_video(pixels)
        else:
            continue
        save_name = 'dvideo_' + str(count) + '.npy'
        np.save(os.path.join(save_dir, save_name), video_out)
        count += 1


def de_identify_video(video, blur_size=8, unblur_size=5, eps=10):
    """
    De-identify a video with avg kernel

    Arguments
        video: np array of the video
        blur_size: 
        un_blur_size:
        eps:

    Returns
        video_out: de-identified array. returns None if de-identification fails
    """
    frames, height, width = video.shape
    temp = np.copy(video)
    temp = np.where(temp<5, 0, temp)
    temp = np.sum(temp,0).clip(0,255).astype(np.uint8)

    kernel = np.ones((blur_size, blur_size), np.float32)/(blur_size**2)
    filtered_gray = cv2.filter2D(temp,-1,kernel)
    ret, thresh = cv2.threshold(filtered_gray,250,255,cv2.THRESH_BINARY_INV)
    mask = 1-thresh.clip(0,1)  
    mask[0:height//10,:] = 0

    kernel = np.ones((unblur_size, unblur_size), np.float32)
    filtered_mask = cv2.filter2D(mask,-1,kernel).clip(0,1)
    filtered_mask = np.where(filtered_mask==0, 0, 1)
    inside_mask = np.where(filtered_mask==1)
    try:
        left_bottom_x = min(inside_mask[1])
        right_bottom_x = max(inside_mask[1])
        top_y = min(inside_mask[0])
        left_top_x = min(inside_mask[1][inside_mask[0]==top_y])
        right_top_x = max(inside_mask[1][inside_mask[0]==top_y])
        delta = blur_size
        left_bottom_x+=delta
        left_top_x-=delta 
        right_top_x+=delta
        right_bottom_x-=delta
        left_bottom_y = min(inside_mask[0][inside_mask[1]==left_bottom_x])
        left_top_y = min(inside_mask[0][inside_mask[1]==left_top_x])
        right_bottom_y = min(inside_mask[0][inside_mask[1]==right_bottom_x])
        right_top_y = min(inside_mask[0][inside_mask[1]==right_top_x])

        left_slope = (left_top_y-left_bottom_y)/(left_top_x-left_bottom_x)
        left_x_intercept = -left_bottom_y/left_slope + left_bottom_x
        leftmost = [left_slope, left_x_intercept]
        right_slope = (right_top_y-right_bottom_y)/(right_top_x-right_bottom_x)
        right_x_intercept = -right_bottom_y/right_slope + right_bottom_x
        rightmost = [right_slope, right_x_intercept]

        m1, m2 = np.meshgrid(np.arange(width), np.arange(height))
        # use epsilon to avoid masking part of the echo
        mask = leftmost[0]*(m1-leftmost[1])-eps<m2 
        mask *= rightmost[0]*(m1-rightmost[1]) - eps <m2 
        mask = np.reshape(mask, (height, width)).astype(np.int8)
        mask[top_y+delta:] = 0
        filtered_mask += mask
    except:
        # print('De-identified failed')
        return None
    filtered_mask = filtered_mask.clip(0,1)
    video_out = video * filtered_mask
    video_out = video_out.astype(np.uint8)
    return video_out