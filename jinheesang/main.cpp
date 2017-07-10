//
//  main.cpp
//  HelloCV
//
//  Created by Huy Nguyen on 5/3/13.
//  Copyright (c) 2013 HelloCv. All rights reserved.
//

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

#define NUM_PLANES 3
#define EPSILON_D 0.4

using namespace std;

void readNextInput(int &curFileNum, cv::Mat &mat);

void readNextInput(int &curFileNum, cv::Mat &mat){
    curFileNum += 1;
    char curFileName[5];
    sprintf(curFileName, "%04d", curFileNum);
    string curFileStr(curFileName);
    
    // get next readed filename
    string filename = "images/" + curFileStr + ".tiff";
    cout << "Opening image = " << filename << endl;
    
    cv::Mat input;
    
    //read image
    input = cv::imread(filename, 0);
    
    //convert uchar to float
    input.convertTo(mat, CV_32FC1);
    
    cout << "Image dimensions = " << mat.size() << endl;
    //cout << rows << " ," << cols << endl;

}

void normalizeMat2d(cv::Mat &mat);
void normalizeMat2d(cv::Mat &mat){
    int rows = mat.rows;
    int cols = mat.cols;
    //normalization (0~1)
    for(int i=0; i<cols; i++){
        for(int j=0; j<rows; j++){
            mat.at<float>(i,j) /= 255;
        }
    }
}

void calculFx(cv::Mat &mat, cv::Mat &matFx);
void calculFx(cv::Mat &mat, cv::Mat &matFx){
    int rows = mat.rows;
    int cols = mat.cols;
    //f(x)
    for(int i=0; i<cols; i++){
        for(int j=0; j<rows; j++){
            if(mat.at<float>(i,j) >= EPSILON_D){
                matFx.at<float>(i,j) = 0;
            }
            else
                matFx.at<float>(i,j) = 1;
        }
    }
}

int main(int argc, const char * argv[]){
    string windowName = "Hello OpenCV";
    cv::Mat mat, matFx;
    vector<cv::Mat> mat3d;
    vector<cv::Mat> mat3dFx;
    cout << "Hello, OpenCV!" << endl;
    
    int curFileNum = 0;
    while(curFileNum < NUM_PLANES){
        //read input image
        readNextInput(curFileNum, mat);
        
        //make mat2d to normalization
        normalizeMat2d(mat);
        mat3d.push_back(mat);
        
        //caculate f(x)
        matFx = cv::Mat(mat.rows,mat.cols, CV_32FC1, float(0));
        calculFx(mat, matFx);
        mat3dFx.push_back(matFx);
        
        // show image in cv window
        // input: original, mat: f(x)
        cv::imshow(windowName, matFx);
        
        // wait for keypress to exit
        cv::waitKey(0);
    }
    return 0;
}

