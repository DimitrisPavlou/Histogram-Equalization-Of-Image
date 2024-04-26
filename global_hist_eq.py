import numpy as np
from helper_functions import equalization_transform
#===================================================================================
 
def perform_global_hist_equalization(image_array : np.ndarray) -> np.ndarray :

    # INPUT 
    # image_array : a 2 dimensional np.ndarray (matrix) that represents an 8-bit gray scale image
    #
    # OUTPUT 
    # image_equalized : the globally equalized image 

    # get the transformation 
    transform = equalization_transform(image_array) 
    # map the values of the original image to the transformed values and cast them to an 8-bit integer
    image_equalized = np.round(transform[image_array.flatten()]).astype(np.uint8)
    # reshape to original shape 
    image_equalized = image_equalized.reshape(image_array.shape) 

    return image_equalized

