# CTimageProcessing
2017 Summer UCI Project - Extracting information from micro CT scans.

## Abstract

It is difficult to produce realistic images for complexible materials. As the materials naturally contain irregularities, it is necessary to do additional work to find the density of materials from CT scans. In this paper, the working procedure is extracting orientation fields and preprocessing the data.
 
The original input of the project is CT imaging. It is can be used to determine the direction and density of each position of the fiber. The direction of each part of the fiber is expressed in RGB color. Each color represents the direction in which the fibers extend. Furthermore, more sophisticated directions can be extracted. To improve direction image, noises are eliminated. The scattered dots are noisy fibers. By this, the researchers only represent actual fibers in RGB. The next result is a density image.  Density indicates how condensed each fiber is. After dividing extracted information by axis, it will be checked as fiber in each coordinate axis. The plane of the separated information is dotted. Method of creating polylines, weighted bipartite matching, is to traverse points on other planes that are closest to one point. Next part is to refine the data through polyline smoothing. After these process, It will be able to model the material that looks like reality through modeling.

## Introduction
Computed microtomography (micro-CT) has been widely used to acquire detailed geometric information from physical samples. The researchers can use the information obtained from micro-CT to analyze the characteristics of various materials. Micro-CT images provide a voxelized density field with no directed information, so the procedure is represented extracting orientation fields and preprocessing the data. In the next step, larger models are rendered using acquired volumetric appearance and geometry models. The resulting renderings show direct information about geometry and produce excellent appearance from the small scale to the large scale. As a result, the goal of this project is to develop software tools that benefit processing methods for fibrous materials. In other words, these software tools can help create more realistic models when modeling complex fibers such as gabardine, silk, velvet, and felt.


## Methods And Materials

### Step 0. Reading 3D-CT images
It takes a long time to read and process a 3D image one by one. So the researchers adopted a way to process 3D images by reading a VOL file containing 3D micro-CT images. In the VOL file, 1 Byte to 48 Byte contains the header information of the corresponding data such as the 3D image size information and channel information, and the rest of the data after 48 Byte includes the actual 3D CT image values. Be careful when reading actual data values. These data values are stored in a little endian system, and the researchers should read the data with this fact in mind. 

### Step 1. Determining Epsilon D
The first thing to do is to extract the binary image from the Micro CT scan images. The CT raw images are a 3D volume of the fabric. This image is grayscale and is represented by a floating point number from 0 to 1.
 	
The binary image is composed of 0 and 1. The fabric is represented by 0 (black) and the background is represented by 1 (white). The following formula f is used to distinguish the background from the fabric.
The appropriate value of epsilon D used in the above equation should be found. Epsilon D is a floating point number from 0 to 1.
If the value is bigger, the fabric will be recognized as the background, using smaller value will result in more noise in the binary image. 

### Step 2. Computing fiber direction Set
To obtain a 3D model of a fabric, It is necessary to know the direction and density of each position of the fabric. The step 2 is to find the direction of each voxel in the fabric. Using the CT raw image and the epsilon D found in step 1, the direction and J value can be found. The J value is determined together when the direction of the voxel is determined. This value will be used in step 3.

1) Getting set of directions.
Before deciding on the voxel direction, It is necessary to define first define the candidate directions. The direction is represented by a unit vector existing in the spherical coordinate system. The larger the theta value and the N value, the greater the number of directions. You can then determine the direction of the voxel more accurately.

2) Finding the J value
To determine the direction of a voxel x, the J value should be determined for all directions obtained in step 2-1. The equation obtaining the J value for voxel x and direction d is as follows.

V is a cube with a length h. In other words, x + p means voxels in a cube centered at x. To find the J value, researchers need to filter the binary image from step 1. The definition of the filter q is as follows.
where r = "p (p · d)d" is the distance from the filter’s axis and the parameters s and t (normally s < t) are empirically adjusted based on the size of the fibers present in the sample.


3) Determining the direction.
Researchers know how to find the J value for one voxel and one direction. From now, researchers need to find the J value for all the set of directions from step 2-1. Researchers need to find the Maximum J value by rotating the filter on the binary image. In this process, the direction d of the filter that generates Maximum J is determined as the direction of voxel x. Researchers used filters to determine the direction of each voxel. The calculated direction is represented by a (x, y, z) direction vector, and researchers express it as an RGB value of 0 to 255.


### Step 3. Determining Epsilon J
Step 3 is to create a higher quality image by clearing the CT raw image. The maximum J value of each voxel obtained in step 2 is used to eliminate the noise. The equation for denoising a CT image is as follows.

Epsilon D is the value already determined in step 1. Therefore, researchers only need to determine the epsilon J value. Researchers need to find the best epsilon J value to make the CT image cleanest. As a result, Denoised image is as follows.

If the epsilon J value is too small, the noise can not be eliminated properly. If the epsilon J value is too large, the fabric is recognized as noise so the fabric is lost.


### Step 4. Dividing Volume
The step 4 is to decompose the directional volume image created in step 2 into three sub-volumes. Fiber structure could be constructed using extracted input density volume. The input volume is supposed to corresponding to each axis, so it is decomposed into three volumes. It’s determined by selecting the component of the direction vector that is largest in absolute value. Each sub-volume consists of voxels with only directional (x, y, z) axes.


### Step 5. Blob detection
The final goal is to implement a 3D fabric model. To do this, researchers need to create a centerline that forms the fabric. And to do this, researchers must find the points that form the line. These points are called Blob. The step 5 is to detect blobs. Then it is easy to extract connected components and calculate fiber centers. Assume that there is a subvolume pointing in the z-axis direction. (step 4) Place a plane perpendicular to the z-direction in this subvolume.

At this time, the point where this plane and the straight line formed by the voxels meet can be obtained as follows.

The 2D slice image converts to the binary images by applying thresholding. After that, researchers can find the blobs in these white dots. A Blob is a group of connected pixels in an image that share some common property. Applying a blob detecting image to this image, researchers can represent the following blob image.



### Step 6. Making Polylines
The final step is to form a polyline, the centerline of the fabric. Suppose, create several 2D slice images created in Step 5. Then several planes are created as follow image. 

Detected fiber centers in adjacent slices are linked together in this step by solving a series of bipartite graph matching problems.  

Finally, when the radius of this polyline is determined, the 3D fabric model is completed.
