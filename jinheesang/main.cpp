//
//  main.cpp
//  HelloCV
//
//  Created by Huy Nguyen on 5/3/13.
//  Copyright (c) 2013 HelloCv. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cmath>
#include "opencv2/opencv.hpp"

#define NUM_PLANES 3
#define EPSILON_D 0.4

#define N_THETA 10
#define DELTA_Z 0.25

typedef struct{
    float x;
    float y;
    float z;
    float theta;
}Direction;

using namespace std;

void readNextInput(int curFileNum, cv::Mat &mat);

void readNextInput(int curFileNum, cv::Mat &mat){
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

float calculJ();
float calculJ(){
    return 0;
}

void calculVoxelDirection(cv::Mat &matFx, cv::Mat &matDir);
void calculVoxelDirection(cv::Mat &matFx, cv::Mat &matDir){
    int rows = matFx.rows;
    int cols = matFx.cols;
    //f(x)
    for(int i=0; i<cols; i++){
        for(int j=0; j<rows; j++){
            if(matFx.at<int>(i,j) == 0){
                float maxJ = 0;
                float tempJ = calculJ();
            }
        }
    }
}


void calculSetOfDirections(vector<Direction> &setOfDirections);
void calculSetOfDirections(vector<Direction> &setOfDirections){
    
    float dividedTheta = 2 * M_PI / N_THETA;
    for(float z=-1; z<=1; z+=DELTA_Z){
        for(int i_theta=0; i_theta <=N_THETA; i_theta += 1){
            float theta = i_theta * dividedTheta;
            float x = (sqrt( 1 - pow(z, 2.0) )) * cos(theta);
            float y = (sqrt( 1 - pow(z, 2.0) )) * sin(theta);
            Direction tempDir = {x, y, z};
            setOfDirections.push_back(tempDir);
        }
    }
}

void showAllDirections(vector<Direction> &setOfDirections){
    for(int i= 0; i<(int)setOfDirections.size(); i++){
        Direction t = setOfDirections[i];
        cout << "x: "<< t.x << "  y: "<< t.y << "  z: "<< t.z << endl;
    }
}

int main(int argc, const char * argv[]){
    string windowName = "Hello OpenCV";
    cv::Mat mat, matFx, matDir;
    
    vector<cv::Mat> mat3d, mat3dFx, mat3dDir;
    vector<Direction> setOfDirections;
    
    
    cout << "Hello, OpenCV!" << endl;

    
    calculSetOfDirections(setOfDirections);
    showAllDirections(setOfDirections);
    
    
    for(int curFileNum = 0; curFileNum < NUM_PLANES; curFileNum++){
        //read input image
        readNextInput(curFileNum, mat);
        
        //make mat2d to normalization
        normalizeMat2d(mat);
        mat3d.push_back(mat);
        
        //caculate f(x)
        matFx = cv::Mat(mat.rows,mat.cols, CV_32FC1, float(0));
        calculFx(mat, matFx);
        mat3dFx.push_back(matFx);
        
        //set directions
        //matDir = cv::Mat(mat.rows, mat.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        //calculDirections(matFx, matDir);
        //mat3dDir.push_back(matDir);
        
        
        // show image in cv window
        // input: original, mat: f(x)
        cv::imshow(windowName, matFx);
        
        // wait for keypress to exit
        cv::waitKey(0);
    }
    return 0;
}

