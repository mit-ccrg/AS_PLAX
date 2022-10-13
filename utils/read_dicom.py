"""
Some utility function to read a dir containing dicoms or a dicom file
"""

import pydicom
import os

def convert_study(study_dir, save_dir, video_format = 'npy', img_format = 'npy'):
    """
    Read all files in an echo study dir and convert the images and videos into desired format.
    
    Arguments
        study_dir: dir of the study
        save_dir: dir to output the converted files
        video_format: npy or avi
        img_format: npy or jpeg

    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pass

def read_study(study_dir):
    """
    Read all files in an echo study dir and return summaries of the study.
    
    Returns
        file_names: list of all names of the files
        dims: list of dimensions of each file. 3 is an image, 4 is a video
        frame_numbers: list of number of frames of each file, 1 indicates the file contains an image
    """
    file_names, dims, frame_numbers = [], [], []
    file_list = os.listdir(study_dir)
    for file in file_list:
        if 'txt' in file:
            continue
        _, pixel_dim, n_frames = read_dicom(os.path.join(study_dir, file))
        file_names.append(file)
        dims.append(pixel_dim)
        frame_numbers.append(n_frames)
    return file_names, dims, frame_numbers

def read_dicom(path, verbose = False):
    """
    open a dicom file and return the pixel array, its dimension and number of frames
    """
    ds = pydicom.dcmread(os.path.join(path),force=True)
    pixels = ds.pixel_array[:,:,:,0]
    pixel_dim = len(pixels.shape)
    if ("NumberOfFrames" in  dir(ds)):
        n_frames = ds.NumberOfFrames
    else:
        n_frames = 1
    return pixels, pixel_dim, n_frames
