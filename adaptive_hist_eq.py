import numpy as np 
from typing import Dict , Tuple
from helper_functions import equalization_transform
#===================================================================================

def adaptive_hist_eq_no_interp(image_array: np.ndarray , region_len_h: int , region_len_w: int) -> np.ndarray : 
    # INPUT 
    # image_array: a 2 dimensional np.ndarray (matrix) that represents an 8-bit gray scale image
    # region_len_h: the height of a region
    # region_len_w: the width of a region
    # 
    # OUTPUT 
    # equalized_image : a 2 dimensional np.ndarray (matrix) that represents the equalized image 
    #                   but computed without interpolating between regions

    (m , n) = image_array.shape
    equalized_image = np.zeros_like(image_array) 
    # raise an AsertionError if the dimensions len_h , len_w provided do not divide m and n respectively 
    assert m % region_len_h == 0 and n % region_len_w == 0 , "The region dimensions must divide the image dimensions"

    # find the contextual regions of the image 
    regions = [(i , j) for i in range(0 , m , region_len_h) for j in range(0 , n , region_len_w)]

    for i in range(len(regions)) : 
        # find center of region i
        start_h = regions[i][0] 
        start_w = regions[i][1]
        # region array in the image 
        region = image_array[start_h: start_h + region_len_h , start_w: start_w + region_len_w]
        # transformation of image
        transform = equalization_transform(region)
        region_equalized = transform[region.flatten()]
        # save the transformation of the region to the corresponding region in the equalized image
        equalized_image[start_h: start_h + region_len_h , start_w: start_w + region_len_w] = \
            region_equalized.reshape(region_len_h , region_len_w)

    return equalized_image.astype(np.uint8)

#===================================================================================

def get_equalization_transform_of_regions(image_array: np.ndarray , region_len_h: int , region_len_w: int)-> Dict[Tuple , np.ndarray]:
    # INPUT 
    # image_array: a 2 dimensional np.ndarray (matrix) that represents an 8-bit gray scale image
    # len_h: the height of a region
    # len_w: the width of a region
    #
    # OUTPUT 
    # transforms_dictionary : a dictionary that has as keys the centers the contextual regions of the image
    #                         and as values the corresponding equalization transform of the region
    

    #find the regions and their centers
    (m , n) = image_array.shape
    # raise an AsertionError if the dimensions len_h , len_w provided do not divide m and n respectively 
    assert (m % region_len_h == 0 and n % region_len_w == 0) , "The region dimensions must divide the image dimensions"

    # find the contextual regions of the image 
    regions = [(i , j) for i in range(0 , m , region_len_h) for j in range(0 , n , region_len_w)]
    k = len(regions)
    transforms_dictionary = {}
    for i in range(k) : 
        start_h = regions[i][0]
        start_w = regions[i][1]
        # find center 
        center_h = start_h + region_len_h//2
        center_w = start_w + region_len_w//2
        # find the transformation of the region
        transform = equalization_transform(image_array[start_h: start_h + region_len_h, start_w: start_w + region_len_w])
        transforms_dictionary[(center_h , center_w)] = transform

    return transforms_dictionary

#===================================================================================

def perform_adaptive_equalization_transform(image_array: np.ndarray , region_len_h: int , region_len_w: int)-> np.ndarray:
    # INPUT 
    # image_array: a 2 dimensional np.ndarray (matrix) that represents an 8-bit gray scale image
    # len_h: the height of a region
    # len_w: the width of a region
    #
    # OUTPUT 
    # equalized_image : a 2 dimensional np.ndarray (matrix) that represents the equalized image

    (height , width) = image_array.shape

    assert height % region_len_h == 0 and width % region_len_w == 0 , "The region dimensions must divide the image dimensions"
    # number of regions in each row 
    num_regions_horizontal = width//region_len_w  

    # find the transformations of each region and the centers
    transforms_dict = get_equalization_transform_of_regions(image_array , region_len_h , region_len_w)
    centers = list(transforms_dict.keys())

    # initialize equalized image
    equalized_image = np.zeros_like(image_array)
    # initialize counters that help track the region we are in 
    region_counter = 0    
    row_counter = 0

    for i in range(height) : 
        for j in range(width) : 
            # if we reach the end of a region update the region counter
            if j!= 0 and j % region_len_w == 0 and j < width: 
                region_counter += 1
              
            # pixel_value 
            pixel = image_array[i,j]
            # find region center 
            center_h , center_w = centers[region_counter]
            # check if pixel is the center 
            if i == center_h and j == center_w : 
                equalized_image[i , j] = transforms_dict[(center_h , center_w)][pixel]
            
            # check if pixel is boundary 
            elif (i <= region_len_h//2 or i >= height - region_len_h//2) or (j <= region_len_w//2 or j >= width - region_len_w//2) : 
                equalized_image[i , j] = transforms_dict[(center_h , center_w)][pixel]
            
            # the pixel is an inner pixel
            else :
                # check quadrant 
                
                # we use <= , >= symbols and not < , > sumbols to consider the cases in which
                # the pixel is collinear with its center 
                # it's worth noting that the case i == center_h and j == center_w is never true at this  
                # stage , given that it has already been covered by the first if statement. Only one equality 
                # can be true , hence covering the boundary cases
                
                if i <= center_h and j <= center_w : 
                    # upper left quadrant of region 
                    center11 = (center_h, center_w - region_len_w) # left center
                    center12 = (center_h - region_len_h , center_w - region_len_w) #upper left center
                    center21 = (center_h , center_w) #original center
                    center22 = (center_h - region_len_h, center_w) #upper center
                    
                if i <= center_h and j >= center_w :
                    # upper right quadrant of region 
                    center11 = (center_h , center_w) #original center
                    center12 = (center_h - region_len_h, center_w) #upper center
                    center21 = (center_h  , center_w + region_len_w) # right center
                    center22 = (center_h - region_len_h , center_w + region_len_w) #upper right center
                    
                if i >= center_h and j <= center_w :
                    # lower left quadrant of region 
                    center11 = (center_h + region_len_h , center_w - region_len_w) #lower left center
                    center12 = (center_h  , center_w - region_len_w) # left center
                    center21 = (center_h + region_len_h, center_w) #down center
                    center22 = (center_h , center_w) #original center

                if i >= center_h and j >= center_w :
                    # lower right quadrant of region 
                    center11 = (center_h + region_len_h, center_w) #down center
                    center12 = (center_h , center_w) #original center
                    center21 = (center_h + region_len_h , center_w + region_len_w) #bottom right center
                    center22 = (center_h  , center_w + region_len_w) # right center

                T11 = transforms_dict[center11][pixel]
                T12 = transforms_dict[center12][pixel]
                T21 = transforms_dict[center21][pixel]
                T22 = transforms_dict[center22][pixel]
                temp = T11*(center22[1]-j)*(center22[0]-i) + T21*(j - center11[1])*(center22[0] - i) + \
                      T12*(center22[1] - j)*(i - center11[0]) + T22*(j - center11[1])*(i - center11[0]) 
                equalized_image[i , j] = np.round((1/((center22[0] - center11[0])*(center22[1]-center11[1]))) * temp)
                
        # make the region counter equal to the row_counter so we can continue studying the row_counter row
        # of the regions list

        region_counter = row_counter
        # if we are done with the first row of regions , move on to the next one
        if i > 0 and i % region_len_h == 0 and i != height - 1 : 
            row_counter += num_regions_horizontal
            region_counter = row_counter
      
    return equalized_image.astype(np.uint8)