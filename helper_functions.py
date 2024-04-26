import numpy as np


#===================================================================================

def image_histogram(image_array : np.ndarray)-> np.ndarray: 
    # INPUT 
    # image_array : a 2 dimensional np.ndarray (matrix) that represents an 8-bit gray scale image
    #
    # OUTPUT 
    # histogram : a 1-d np.ndarray with the frequencies of the pixel values of each pixel of the input image

    #reate empty histogram array
    L = 256 
    histogram = np.zeros(L)

    for pixel in image_array.flatten():
        #pixel is in the range [0 , 255]
        # increament the count of the pixel value in the image by one 
        histogram[pixel] += 1
    
    return histogram

#===================================================================================

def cumsum(array: np.ndarray)-> np.ndarray:
    
    # INPUT
    # array : an 1-d np.ndarray 
    #
    # OUTPUT 
    # b : an 1-d np.ndarray with the i-th element being the
    #     cumulative sum up to the i-th element of the input array
    
    #convert np.ndarray to iterable
    array = iter(array)
    b = [next(array)]
    
    for element in array:
        b.append(b[-1] + element)
    
    return np.array(b)

#===================================================================================

def equalization_transform(image_array : np.ndarray) -> np.ndarray: 

    # INPUT 
    # image_array : a 2 dimensional np.ndarray (matrix) that represents an 8-bit gray scale image
    #
    # OUTPUT 
    # equalization_transform : the transformation of the image

    L = 256
    (m , n) = image_array.shape
    #find the frequencies of the pixel values in the image
    hist = image_histogram(image_array)
    #normalize the histogram by dividing with the total number of pixels
    hist = hist/(m*n)
    #find the cdf 
    cdf = cumsum(hist) 
    #apply the transformation (element-wise)
    equalization_transform = np.round(((cdf - cdf[0])/(1 - cdf[0]))*(L-1))

    return equalization_transform


