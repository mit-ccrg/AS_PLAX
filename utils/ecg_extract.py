import numpy as np
import skimage.measure
import skimage.morphology

def get_largest_connected_area(segmentation):
    labels = skimage.measure.label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def threshold_by_color_distance(
    image: np.ndarray, 
    color: np.ndarray, 
    distance: float
):
    """Applies a threshold in RGB color space

    Arguments
        image: numpy array of the RGB color image. Expected shape: (nrows x ncols x 3)
        color: numpy array of the ECG waveform color. Expected shape: (3,)
        distance: threshold distance in pixel units
    
    Returns
        a binary numpy array of the thresholded image
    """
    return np.linalg.norm((image - color), axis=2) < distance


def extract_ecg_box(
    image: np.ndarray, 
    color: np.ndarray, 
    distance: float
):    
    """Extracts the bounding box of the ECG waveform

    Arguments
        image: numpy array of the RGB color image. Expected shape: (nrows x ncols x 3)
        color: numpy array of the ECG waveform color. Expected shape: (3,)
        distance: threshold distance in pixel units
    
    Returns
        A tuple of four integers indicating the left and right column numbers; and the top 
        and bottom row numbers bounding the ECG waveform
    """
    ecg_pixels = threshold_by_color_distance(image, color, distance)
    row_sum = ecg_pixels.sum(axis=0)
    col_sum = ecg_pixels.sum(axis=1)
    
    left_ecg = np.argwhere(row_sum).min()
    right_ecg = np.argwhere(row_sum).max()
    top_ecg = np.argwhere(col_sum).min()
    bottom_ecg = np.argwhere(col_sum).max()
    
    return left_ecg, right_ecg, top_ecg, bottom_ecg


def extract_ecg_image(
    image: np.ndarray, 
    color: np.ndarray, 
    distance: float,
    distance_factor: float, 
    border: int,     
    close: bool,
    close_kernel: int,
    skeleton: bool, 
    largest: bool
):
    """
    Extracts the ECG waveform from an echo image frame

    Arguments:
        image: numpy array of the RGB color image. Expected shape: (nrows x ncols x 3)
        color: numpy array of the ECG waveform color. Expected shape: (3,)
        distance: threshold distance in pixel units
        distance_factor: multiplicative factor used to extract the waveform after cropping
        border: number of border pixels to include around the waveform
        close: performs a horizontal morphological closing (helpful if the waveform is disconnected)
        close_kernel: width (in pixel) of the kernel used for the morphological closing
        skeleton: skeletonize the waveform (helpful to have a single-pixel waveform)
        largest: extracts only the largest connected waveform (helpful if image is noisy)

    Returns:
        Numpy array containing a binary image with the extracted ECG waveform
    """
    left, right, top, bottom = extract_ecg_box(image, color, distance)
    ecg_pixels = threshold_by_color_distance(image, color, distance*distance_factor)
    ecg_pixels = ecg_pixels[top-border:bottom+border, left-border:right+border]
    if close:
        ecg_pixels = skimage.morphology.closing(ecg_pixels, np.ones((1, close_kernel)))
    if skeleton:
        ecg_pixels = skimage.morphology.skeletonize(ecg_pixels, method='lee')
    if largest:
        ecg_pixels = get_largest_connected_area(ecg_pixels)
    return ecg_pixels
    

def get_ecg_progress(
    ecg_image: np.ndarray,    
):
    """Extracts rightmost pixel from an ECG waveform image
    Arguments:
        ecg_image: numpy array of the ECG waveform binary image
    Returns:
        column number of the right end of the ECG waveform
    """
    if len(ecg_image) == 0:
        return 0
    row_sum = ecg_image.sum(axis=0)
    right_ecg = np.argwhere(row_sum).max()
    return right_ecg


def extract_ecg_video(
    video: np.ndarray,
    color: np.ndarray = np.array([174, 65, 52]),
    distance: float = 20.0,
    distance_factor: float = 5.0, 
    border: int = 0,     
    close: bool = False,
    close_kernel: int = 3,
    skeleton: bool = False, 
    largest: bool = False
):
    """Extracts the ECG waveform from an echo movie as well as column coordinates of the waveform at each frame

        Arguments:
            video: numpy array of the echo movie. Expected shape: (nframes x nrows x ncols x 3)
            color: numpy array of the ECG waveform color. Expected shape: (3,)
            distance: threshold distance in pixel units
            distance_factor: multiplicative factor used to extract the waveform after cropping
            border: number of border pixels to include around the waveform
            close: performs a horizontal morphological closing (helpful if the waveform is disconnected)
            close_kernel: width (in pixel) of the kernel used for the morphological closing
            skeleton: skeletonize the waveform (helpful to have a single-pixel waveform)
            largest: extracts only the largest connected waveform (helpful if image is noisy)
        Returns:
            Numpy array of the binarized image containing the extracted ECG waveform
            List of the column coordinates indicating waveform progress at each frame
    """
    ecg_end = extract_ecg_image(
        video[-1],
        color, 
        distance, 
        distance_factor,
        border,
        close,
        close_kernel,        
        skeleton, 
        largest
    )
    nframes = video.shape[0]

    progress = []
    for i in range(nframes):
        if i<3:
            try:
                ecg_i = extract_ecg_image(
                    video[i],
                    color, 
                    distance, 
                    distance_factor,
                    border,
                    close,
                    close_kernel,        
                    skeleton, 
                    largest
                )
                progress.append(get_ecg_progress(ecg_i))
            except:
                progress.append(0)
        else:
            ecg_i = extract_ecg_image(
                video[i],
                color, 
                distance, 
                distance_factor,
                border,
                close,
                close_kernel,        
                skeleton, 
                largest
            )
            progress.append(get_ecg_progress(ecg_i))
    return ecg_end.astype('int'), np.array(progress)