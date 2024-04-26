from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from global_hist_eq import * 
from adaptive_hist_eq import *
from helper_functions import *
#===================================================================================
#a list of numbers from 0 to 255 representing the bins of the histograms that we 
#will plot 
BINS = range(0 , 256)

#===================================================================================

#image file
filename = "input_img.png"
img = Image.open(fp=filename)
#convert to gray scale
bw_image = img.convert("L")
#convert image to a a numpy array of 8bit integers
image_array = np.array(bw_image).astype(np.uint8)
#===================================================================================

#ORIGINAL IMAGE

#bins of histogram 
#num_bins = 256 #BINS = range(0 , 256)

#compute histogram
counts_original = image_histogram(image_array)

#plot the original image and its histogram 
plt.figure(1)
plt.imshow(image_array , cmap = "gray")
plt.axis("off")
plt.title("Original image")
plt.figure(2)
plt.bar(BINS , counts_original, width=1)
plt.grid()
plt.title("Histogram of original image")

#===================================================================================

#GLOBAL HISTOGRAM EQUALIZATION

#global equalization transformation 
equalized_image_global = perform_global_hist_equalization(image_array)
#histogram of globally equalized image
counts_global = image_histogram(equalized_image_global) 

#plot globally equalized image
plt.figure(3)
plt.imshow(equalized_image_global , cmap = "gray")
plt.axis("off")
plt.title("Globally equalized image")
#plot histogram
plt.figure(4)
plt.bar(BINS , counts_global , width=1)
plt.grid()
plt.title("Histogram of globally equalized image")

#plot equalization transform
plt.figure(5)
eq_transform = equalization_transform(image_array)
plt.step(BINS , eq_transform)
plt.grid()
plt.title("Equalization transform")
plt.savefig("eq_transform.png")
#===================================================================================

#ADAPTIVE HISTOGRAM EQUALIZATION

#region height for adaptive histogram equalization
LEN_H = 60
#region width for adaptive histogram equalization
LEN_W = 48


#adaptively equalized image with regions of size len_h x len_w
equalized_image_adaptive = perform_adaptive_equalization_transform(image_array, LEN_H, LEN_W)
#histogram of adaptively equalized image 
counts_adaptive = image_histogram(equalized_image_adaptive)

plt.figure(6)
plt.imshow(equalized_image_adaptive , cmap = "gray")
plt.axis("off")
plt.title(f"Adaptively equalized image with region size {LEN_H}x{LEN_W}")
#plot histogram
plt.figure(7)
plt.bar(BINS , counts_adaptive, width=1)
plt.grid()
plt.title(f"Histogram of adaptively equalized image with region size {LEN_H}x{LEN_W}")

#===================================================================================
#
#ADAPTIVE HISTOGRAM EQUALIZATION WITHOUT INTERPOLATION

equalized_image_adaptive_no_interp = adaptive_hist_eq_no_interp(image_array , LEN_H , LEN_W)
#histogram of adaptively equalized image 
counts_adaptive_no_interp = image_histogram(equalized_image_adaptive_no_interp)

plt.figure(8)
plt.imshow(equalized_image_adaptive_no_interp , cmap = "gray")
plt.axis("off")
plt.title(f"Adaptively equalized image (no interpolation) with region size {LEN_H}x{LEN_W}")
#plot histogram
plt.figure(9)
plt.bar(BINS , counts_adaptive_no_interp, width=1)
plt.grid()
plt.title(f"Histogram of adaptively equalized image (no interpolation) with region size {LEN_H}x{LEN_W}")


plt.show()


