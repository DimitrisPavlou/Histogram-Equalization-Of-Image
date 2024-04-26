# Histogram-Equalization-Of-Image
#
Implemented the classic algorithm for global histogram equalization and also the algorithm for adaptive histogram equalization. 
One note for the ahe algorithm is that I have not (yet) implemented the border cases so some discontinuities are obvious between inner contextual regions and border contextual regions are obvious.
I have also provided the test images I used to check the validity of the algorithms. In the demo.py file I use the input_img.png image , but you can change it. Be carefull of the dimensions of the images and the contextual regions. The ahe algorithm assumes that the dimensions of the contextual regions divide perfectly the dimensions of the original image. If thats not the case , an Assertion Exception will be raised prompting to change the dimensions.
The algorithms are writen in Python using numpy.
