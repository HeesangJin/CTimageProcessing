# CTimageProcessing
## Introduction
This program is a program for processing CT images.
The implementation theory is summarized in Extracting Useful Information from Micro-CT Scans of Fibrous Materials.
This program covers Step.0 ~ Step.3 of the above paper.
The program for Step.4 ~ Step.6 is here<Link>.


## Input
The input file of this program is a VOL file.

## Output
The output format of this program is as follows:

1. Original CT image
2. Binary CT image
3. Orientation RGB image
4. Cleaned CT image

By default, the four images are processed and output in order.

### 1. Original CT image
This image is a VOL format input file that is output as an image of Gray Scale.
In other words, the input file itself.

### 2. Binary CT image
This image is an image of the CT image only in black (fabric) and white (background) according to the threshold Epsilon D.
The user can adjust the Epsilon D and see the result.

### 3. Orientation RGB image
This image is the image of the direction of each voxel in the CT image in RGB color.
The x-axis direction is represented by Red, the y-axis direction is represented by Green, and the z-axis direction is represented by Blue. 
Each value represents a (x, y, z) direction vector.
<Note> This processing takes a very long time.

### 4.Cleaned CT image
This image is a CT image that has been refined to a cleaner form by eliminating the noise of the original CT image according to the threshold Epsilon J.
The user can adjust the Epsilon J and see the result.


## Usage

### Files
This project contains four files:
1. main.cpp
2. read_VOL.cpp
3. read_VOL.h
4. Makefile

You also need the following VOL files:
1. Pramook_black_velvet_3.03um_80kV_down.vol
<Link>

The above five files must be in the same directory.

### Compile
To run the program, you should compile it as follows:
```
$ make
```

### Run
Run the program as following format:
```
$ ./main TYPE IMAGE_NUMBER
```

Valid values ​​for TYPE and IMAGE_NUMBER are:

- TYPE:

1. 'o': Output Original CT Image.

2. 'b': Binary CT Image is output.

3. 's': Original, Binary Omits the CT Images output.

<Note> Direction RGB Image, Cleaned CT Image are always printed.



- IMAGE_NUMBER:

(Integer) 0 ~ Size of Z: RGB, Cleand Image Number.

<Note> If you do not enter this value, the program will exit without outputting Direction RGB Image and Cleaned CT Image.


## Example

```
$ ./main b 100
```
Outputs a binary CT image for all planes, and outputs the 100th plane Direction RGB Image and Cleaned CT Image.

```
$ ./main s 102
```
Output the Direction RGB Image and Cleaned CT Image of the 102nd plane.

```
$ ./main o
```
Outputs Original CT image for all planes.

```
$ ./main s
```
Nothing is output.
<Note> The image number must be inputed in the second parameter.
