# inf8702
Seamless cloning using Poisson image blending.

##Installation

###C++ requirements
- C++14
- OpenCV
- Boost
- TBB
- OpenCL

You can then use the CMakeList. 

###Python requirements
Only needed to use the MaskMaker.py script to help making masks and resized source images. 

- Python 2.7
- Pillow
- tkinter

##Steps to use
1. Find two images to blend. We refer to the image from which we'll take the patch to blend as the source image, and the image upon witch the patch will be blended as the destination image. 
2. Create a mask of the same size as the destination image indicating in white the regions where the patch will be blended. Also, the source image should be adjusted to be of the same size as the destination image, and aligned so that the mask would take the appropriate patch. In order to do this, we have made a python tool. See bellow the MaskMaker usage section.
3. Build the c++ program (either using cmake's make, or using the xcode project -- mac users only for the later).
4. Run the program, giving it appropriate arguments : `./build/inf8702 destination_image_name source_image_name mask_image_name mixing_gradient_option`, where mixing-gradient-option is 0 for no gradient mixing, 1 for classic maximum based gradient mixing and 2 is for average based gradient mixing. 
5. Open the resulting images called `result-serial-[nb_iterations]-[mixing_gradient_option]`, `result-tbb-[nb_iterations]-[mixing_gradient_option]` and `result-cl-[nb_iterations]-[mixing_gradient_option]`.

##MaskMaker Usage
1. Launch the tool : `python MaskMaker.py`.
2. Create a session using the menu : file -> New Session. Then pick a source image and a destination. Note that both need to be of the same color format (ex : 2 RGB images or 2 RGBA images. Using 1 RGB and 1 RGBA currently doesn't work). Hit Confirm.
3. Using the command described at the top of the tool's window, position the source image and mask (compared to the destination image). Then draw the mask. Note that the mask is transparently blended on the source and destination views.
4. Once the mask and position is to your taste, save the mask and source image using the file menu. Those images should be good to be used by the main c++ program together with the destination given to the tool. 
