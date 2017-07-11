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
#include <climits>
#include "opencv2/opencv.hpp"

#define NUM_PLANES 200
#define EPSILON_D 0.4

#define N_THETA 10
#define DELTA_Z 0.1
#define LENGTH_L 6

#define VALUE_S 3
#define VALUE_T 4

typedef struct{
    float x;
    float y;
    float z;
    float theta;
}Direction;

typedef struct{
    int x;
    int y;
    int z;
}Point;

using namespace std;


vector<cv::Mat> mat3dFx;

int ROWS, COLS;
int num_d;

//vector[num_d][LENGTH_L+1]


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

    //have to be refactoring
    ROWS = rows;
    COLS = cols;

    
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


float calculR(const Direction &d, const Point &p);
float calculR(const Direction &d, const Point &p){
    // r = || p - (p . d) d ||
    float pd = (d.x * p.x) + (d.y * p.y) + (d.z * p.z);
    Direction pdd = { pd * d.x , pd * d.y * pd * d.z };
    Direction ppdd = { p.x - pdd.x , p.y - pdd.y , p.z - pdd.z };
    float r = sqrt( (ppdd.x * ppdd.x) + (ppdd.y * ppdd.y) + (ppdd.z * ppdd.z) );
    
    return r;
}


float calculQ(const Direction &d, const Point &curP);
float calculQ(const Direction &d, const Point &curP){
    float r = calculR(d, curP);
    
    float s = VALUE_S;
    float t = VALUE_T;
    
    //cout << "r is: " << r << ", q is: " << ( -2.0 * exp( -s * r * r ) ) + ( exp( -t * r * r ) ) << endl;
    return ( -2.0 * exp( -s * r * r ) ) + ( exp( -t * r * r ) );
}

//void calculAllQs(vector<Direction> &setOfDirections){
//    for(int d_i=0; d_i<(int)setOfDirections.size(); d_i++){
//        vector<float> temp;
//        for(int r_i=0; r_i<ROWS; r_i++){
//            for(int c_i=0; c_i<COLS; c_i++){
//                for(int p_i=0; p_i<NUM_PLANES; p_i++){
//                    Point curP = {r_i, c_i, p_i};
//                    temp.push_back( calculQ(setOfDirections[d_i], curP) );
//                
//                }
//            }
//        }
//        qValues.push_back(temp);
//    }
//}


bool isValidPoint(Point point);
bool isValidPoint(Point point){
    if(point.x < 0 || point.x >= COLS || point.y < 0 || point.y >= ROWS || point.z < 0 || point.z >= NUM_PLANES)
        return false;
    return true;
}


float getFxFromVoxel(Point curXsumP);
float getFxFromVoxel(Point curXsumP){
    return (mat3dFx[curXsumP.z]).at<float>(curXsumP.x, curXsumP.y);
}

float calculJ(Point &curVoxel, Direction &d);
float calculJ(Point &curVoxel, Direction &d){
    float J=0;
    
    //each p in V(V`s size is l)
    int half_l = LENGTH_L/2;
    for(int p_x= -half_l; p_x<=half_l; p_x++){
        for(int p_y= -half_l; p_y<=half_l; p_y++){
            for(int p_z= -half_l; p_z<=half_l; p_z++){
                Point curP = {p_x, p_y, p_z};
                Point curXsumP = {p_x + curVoxel.x , p_y + curVoxel.y , p_z + curVoxel.z };
                
                // J += f(x+p) * q(d;p)
                if( isValidPoint(curXsumP) ){
                    //cout << "curXsumP.x: "<< curXsumP.x << ", curXsumP.y: "<< curXsumP.y << ", curXsumP.z: " << curXsumP.z <<endl;
                    J += getFxFromVoxel(curXsumP) * calculQ(d, curP);
                    //cout << "current J is: " << J << endl;
                }
            }
        }
    }
    //cout << "return j " << endl;
    return J;
}
            
            
void calculVoxelDirection(Point &curPoint, cv::Mat &matFx, cv::Mat &matDir, vector<Direction> &setOfDirections);
void calculVoxelDirection(Point &curPoint, cv::Mat &matFx, cv::Mat &matDir, vector<Direction> &setOfDirections){
    int rows = matFx.rows;
    int cols = matFx.cols;

    //for each Voxel x
    for(int i=0; i<500; i++){ //cols
        curPoint.x = i;
        for(int j=0; j<500; j++){ //rows
            curPoint.y = j;
            
            // if f(x) = 0
            if(matFx.at<float>(i,j) == 0){
                
                // Calculate Direction(x)
                // find maxJ
                float maxJ = (float)INT_MIN;
                
                //for each d in setOfDirections
                for(int dir_i=0; dir_i<(int)setOfDirections.size(); dir_i++){
                    Direction d = setOfDirections[dir_i];
                    
                    //cout << "i: "<< i << ", y: " << j << ", dir_i: "<< dir_i << endl;
                    //Calculate J(x,d)
                    float tempJ = calculJ(curPoint, d);
                    //cout << "maxJ: " << maxJ << ", tempJ: " << tempJ << endl;
                    if(tempJ > maxJ){
                        maxJ = tempJ;
                        //matDir.r/g/b = d.x/d.y/d.z;
                        
                        //cout << "d.x: "<< abs(d.x) << "d.y: "<< abs(d.y) << "d.z: "<< abs(d.z) << endl;
                        int color_r = abs(d.x) * 255;
                        int color_g = abs(d.y) * 255;
                        int color_b = abs(d.z) * 255;
                        
                        matDir.at<cv::Vec3b>(i,j)[0] = color_b;
                        matDir.at<cv::Vec3b>(i,j)[1] = color_g;
                        matDir.at<cv::Vec3b>(i,j)[2] = color_r;
                    }
                    
                }
                int color_b = matDir.at<cv::Vec3b>(i,j)[2];
                int color_g = matDir.at<cv::Vec3b>(i,j)[1];
                int color_r = matDir.at<cv::Vec3b>(i,j)[0];
                cout << "calcul direction x: " << curPoint.x << ", y: " << curPoint.y << ", z: " << curPoint.z << endl;
                cout << "R: " << color_r << ", G: " << color_g << ", B: " << color_b << endl;
                
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
            float x = (sqrt( 1 - z * z )) * cos(theta);
            float y = (sqrt( 1 - z * z )) * sin(theta);
            Direction tempDir = {x, y, z};
            setOfDirections.push_back(tempDir);
        }
    }
    num_d = (int)setOfDirections.size();
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
    
    vector<cv::Mat> mat3d, mat3dDir;
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
    }
    
    for(int curFileNum = 0; curFileNum < NUM_PLANES; curFileNum++){
        //set directions
        mat = mat3d[curFileNum];
        matFx = mat3dFx[curFileNum];
        cout << "curFileNum: " << curFileNum << endl;
        Point curPoint = {0, 0, curFileNum};
        matDir = cv::Mat(mat.rows, mat.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        calculVoxelDirection(curPoint, matFx, matDir, setOfDirections);
        //mat3dDir.push_back(matDir);
        
        
        // show image in cv window
        // input: original, mat: f(x)
        cv::imshow(windowName, matDir);
        
        // wait for keypress to exit
        cv::waitKey(0);
    }
    return 0;
}

