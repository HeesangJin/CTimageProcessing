//
//  main.cpp
//  HelloCV
//
//  Created by Huy Nguyen on 5/3/13.
//  Copyright (c) 2013 HelloCv. All rights reserved.
//

#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
int main(int argc, const char * argv[])
{
    string filename = "images/0001.tiff";
    string windowName = "Hello OpenCV";
    cv::Mat mat, input;
    
    cout << "Hello, OpenCV!" << endl;
    cout << "Opening image = " << filename << endl;
    
    //read image
    mat = cv::imread(filename, 0);

    //convert uchar to float
    mat.convertTo(input, CV_32FC1);
    
    cout << "Image dimensions = " << mat.size() << endl;
    
    int rows = mat.rows;
    int cols = mat.cols;
    
    cout << rows << " ," << cols << endl;
    
    //set of d in 32x32 cube map.
    //Take all the points above the x-axis and make them into unit vectors.
    
    
    //make 3d points
    int dims[] = {1010, 988, 990};
    cv::Mat mnd(3, dims, CV_64F);
    
    
    //normalization (0~1)
    for(int i=0; i<cols; i++){
        for(int j=0; j<rows; j++){
            input.at<float>(i,j) /= 255;
        }
    }
    
    //f(x)
    for(int i=0; i<cols; i++){
        for(int j=0; j<rows; j++){
            if(input.at<float>(i,j) >= 0.4){
                input.at<float>(i,j) = 0;
            }
            else
                input.at<float>(i,j) = 1;
        }
    }
    //cout << mat.at<uchar>(500,500) << endl;
    
    // show image in cv window
    cv::imshow(windowName, input);
    
    // wait for keypress to exit
    cv::waitKey(0);
    
    return 0;
}

