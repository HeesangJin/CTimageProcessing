//
//  main.cpp
//  HelloCV
//
//  Created by Huy Nguyen on 5/3/13.
//  Copyright (c) 2013 HelloCv. All rights reserved.
//
#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <cmath>
#include <climits>
#include <ctime>

#include "read_VOL.h"
#include "opencv2/opencv.hpp"

#define NUM_PLANES 150
float EPSILON_D = 0.57;
int EPSILON_D_BASE = 0;
float EPSILON_J = -1;
int EPSILON_J_BASE = 0;
float EPSILON_I = 0.7;
int EPSILON_I_BASE = 0;

#define N_THETA 8
#define N_Z 6
#define LENGTH_L 6

#define VALUE_S 3
#define VALUE_T 4

typedef struct{
    float x;
    float y;
    float z;
}Direction;

using namespace std;


vector<cv::Mat> mat3dFx, mat3dCT, mat3dJ;

int ROWS, COLS;
int num_d;

//vector[num_d][LENGTH_L+1]
vector<vector<vector<vector<float> > > > qValues;

string type2str(int type);
void readNextInput(int curFileNum, cv::Mat &mat, vector<float> &data);
void readNextInput(int curFileNum, cv::Mat &mat, vector<float> &data) {
    cv::Mat input(sy, sx, CV_32FC1);

    //read image
    for (int i = 0; i < sy; i++) {
        for (int j = 0; j < sx; j++) {
            Point position = { j,i,curFileNum };
            input.at<float>(i, j) = findData(data, position);
        }
    }
    mat = input;

    cout << "Image dimensions = " << mat.size() << endl;
    //cout << mat.rows << " ," << mat.cols << endl;
}

void readNextInputCH3(int curFileNum, cv::Mat &mat, vector<float> &data);
void readNextInputCH3(int curFileNum, cv::Mat &mat, vector<float> &data) {
    cv::Mat input(sy, sx, CV_32FC3);

    //read image
    for (int i = 0; i < sy; i++) {
        for (int j = 0; j < sx; j++) {
            cv::Point3i position;
            position.x = j;
            position.y = i;
            position.z = curFileNum;
            cv::Point3i rgbData;
            rgbData = findRgbData(data, position);
            input.at<cv::Vec3f>(i, j)[0] = (float)rgbData.x; // r
            input.at<cv::Vec3f>(i, j)[1] = (float)rgbData.y; // g
            input.at<cv::Vec3f>(i, j)[2] = (float)rgbData.z; // b
        }
    }
    mat = input;

    cout << curFileNum <<" Image dimensions = "<< mat.size() << endl;
    //cout << mat.rows << " ," << mat.cols << endl;
}

string type2str(int type) {
    string r;
    
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    
    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }
    
    r += "C";
    r += (chans+'0');
    
    return r;
}

void normalizeMat2d(cv::Mat &mat);
void normalizeMat2d(cv::Mat &mat){
    int rows = mat.rows;
    int cols = mat.cols;
    
    //have to be refactoring
    ROWS = rows;
    COLS = cols;
    
    
    //normalization (0~1)
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            mat.at<float>(i,j) /= 255;
        }
    }
}

void calculFx(cv::Mat &mat, cv::Mat &matFx);
void calculFx(cv::Mat &mat, cv::Mat &matFx){
    int rows = mat.rows;
    int cols = mat.cols;
    
    //f(x)
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            if(mat.at<float>(i,j) >= EPSILON_D){
                matFx.at<float>(i,j) = 0;
            }
            else
                matFx.at<float>(i,j) = 1;
        }
    }
}

void debugToRgb(cv::Mat &matPrev, cv::Mat &matRGB);
void debugToRgb(cv::Mat &matPrev, cv::Mat &matRGB){
    int rows = matPrev.rows;
    int cols = matPrev.cols;
    
    //f(x)
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            if(matPrev.at<float>(i,j) == 1){
                matRGB.at<cv::Vec3b>(i,j)[0] = 0;
                matRGB.at<cv::Vec3b>(i,j)[1] = 0;
                matRGB.at<cv::Vec3b>(i,j)[2] = 0;
            }
            else{
                matRGB.at<cv::Vec3b>(i,j)[0] = 255;
                matRGB.at<cv::Vec3b>(i,j)[1] = 255;
                matRGB.at<cv::Vec3b>(i,j)[2] = 255;
            }
        }
    }
}


float calculR(const Direction &d, const Point &p);
float calculR(const Direction &d, const Point &p){
    // r = || p - (p . d) d ||
    float pd = (d.x * p.x) + (d.y * p.y) + (d.z * p.z);
    Direction pdd = { pd * d.x , pd * d.y, pd * d.z };
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

void calculAllQs(vector<Direction> &setOfDirections){
    int half_l = LENGTH_L/2;
    int numOfL = LENGTH_L;
    if( LENGTH_L%2 == 0) numOfL++;
    vector<vector<vector<vector<float> > > > tempQ(num_d, vector<vector<vector<float> > >
                                                   (numOfL, vector<vector<float> >
                                                    (numOfL, vector<float>
                                                     (numOfL))));
    
    for(int dir_i=0; dir_i<(int)setOfDirections.size(); dir_i++){
        for(int p_x= -half_l; p_x<=half_l; p_x++){
            for(int p_y= -half_l; p_y<=half_l; p_y++){
                for(int p_z= -half_l; p_z<=half_l; p_z++){
                    Point curP = {p_x, p_y, p_z};
                    
                    tempQ[dir_i][p_x+half_l][p_y+half_l][p_z+half_l] = calculQ(setOfDirections[dir_i], curP);
                    
                    //cout << "d: " << setOfDirections[dir_i].x << ", "<< setOfDirections[dir_i].y << ", " <<setOfDirections[dir_i].z;
                    //cout << "///p: " <<curP.x << ", "<< curP.y << ", "<< curP.z;
                    //cout << "///q: " << tempQ[dir_i][p_x+half_l][p_y+half_l][p_z+half_l] << endl;
                }
            }
        }
        
    }
    qValues = tempQ;
}


bool isValidPoint(Direction point);
bool isValidPoint(Direction point){
    if(point.x < 0 || point.x >= COLS || point.y < 0 || point.y >= ROWS || point.z < 0 || point.z >= NUM_PLANES)
        return false;
    return true;
}


float getFxFromVoxel(Point voxel);
float getFxFromVoxel(Point voxel){
    return (mat3dFx[voxel.z]).at<float>(voxel.y, voxel.x);
}

float getValueFromMatrix(int x, int y, int z, vector<cv::Mat> &mat3d);
float getValueFromMatrix(int x, int y, int z, vector<cv::Mat> &mat3d){
    return (mat3d[z]).at<float>(y, x);
}

//Find value of voxel(float) using nearby voxel(int)
float interpolateOneChanValue(Direction &voxel, vector<cv::Mat> &mat3d);
float interpolateOneChanValue(Direction &voxel, vector<cv::Mat> &mat3d){
    int ix = (int)floor(voxel.x);
    int iy = (int)floor(voxel.y);
    int iz = (int)floor(voxel.z);
    
    //error check
    if(ix < 0 || iy >= COLS-1 || iy < 0 || iy >= ROWS-1 || iz < 0 || iz >= NUM_PLANES-1 )
        return 0;
    
    float fx = voxel.x - (float)ix;
    float fy = voxel.y - (float)iy;
    float fz = voxel.z - (float)iz;
    
    float leftBot = (1-fz) * getValueFromMatrix(ix, iy, iz, mat3d) + fz * getValueFromMatrix(ix, iy, iz+1, mat3d);
    float rightBot = (1-fz) * getValueFromMatrix(ix+1, iy, iz, mat3d) + fz * getValueFromMatrix(ix+1, iy, iz+1, mat3d);
    float leftUp = (1-fz) * getValueFromMatrix(ix, iy+1, iz, mat3d) + fz * getValueFromMatrix(ix, iy+1, iz+1, mat3d);
    float rightUp = (1-fz) * getValueFromMatrix(ix+1, iy+1, iz, mat3d) + fz * getValueFromMatrix(ix+1, iy+1, iz+1, mat3d);
    
    float centerBot = (1-fx) * leftBot + fx * rightBot;
    float centerUp = (1-fx) * leftUp + fx * rightUp;
    
    float result = (1-fy) * centerBot + fy * centerUp;
    
    return result;
}

Direction rotateMatrix(Direction voxel, float degree, char axis);
Direction rotateMatrix(Direction voxel, float degree, char axis){
    float preVoxelTemp[3][1] = {{voxel.x}, {voxel.y}, {voxel.z}};
    cv::Mat preVoxel = cv::Mat(3, 1, CV_32FC1, preVoxelTemp);
    
    cv::Mat rotateMtx;
    if(axis == 'x'){
        float rotateTemp[3][3] = {{1, 0, 0}, {0, cos(degree), -sin(degree)}, {0, sin(degree), cos(degree)}};
        rotateMtx = cv::Mat(3, 3, CV_32FC1, rotateTemp);
    }
    else if(axis == 'y'){
        float rotateTemp[3][3] = {{cos(degree), 0, sin(degree)}, {0, 1, 0}, {-sin(degree), 0, cos(degree)}};
        rotateMtx = cv::Mat(3, 3, CV_32FC1, rotateTemp);
    }
    else{
        float rotateTemp[3][3] = {{cos(degree), -sin(degree), 0}, {sin(degree), cos(degree), 0}, {0, 0, 1}};
        rotateMtx = cv::Mat(3, 3, CV_32FC1, rotateTemp);
    }
    
    //cout << "rows: " << preVoxel.rows << ", cols: " << preVoxel.cols << endl;
    cv::Mat resultTemp = rotateMtx * preVoxel;
    Direction result = { resultTemp.at<float>(0,0), resultTemp.at<float>(1,0), resultTemp.at<float>(2,0) };
    return result;
}


Direction rotateVoxelByDegree(int indexDegree, Point voxel, vector<pair<float, float> > &degreesByDirections);
Direction rotateVoxelByDegree(int indexDegree, Point voxel, vector<pair<float, float> > &degreesByDirections){
    pair<float, float> degrees = degreesByDirections[indexDegree];
    
    Direction temp = {voxel.x, voxel.y, voxel.z };
    temp = rotateMatrix(temp, degrees.first, 'x');
    temp = rotateMatrix(temp, degrees.second, 'z');
    
    return temp;
}

float calculJ(Point &curVoxel, Direction &d, int dir_i, vector<pair<float, float> > &degreesByDirections);
float calculJ(Point &curVoxel, Direction &d, int dir_i, vector<pair<float, float> > &degreesByDirections){
    float J=0;
    //each p in V(V`s size is l)
    int half_l = LENGTH_L/2;
    for(int p_x= -half_l; p_x<=half_l; p_x++){
        for(int p_y= -half_l; p_y<=half_l; p_y++){
            for(int p_z= -half_l; p_z<=half_l; p_z++){
                // p*R
                Point curP = {p_x, p_y, p_z};
                Direction curPmulR = rotateVoxelByDegree(dir_i, curP, degreesByDirections);
                
                // p*R+x
                Direction curPmulRsumX = {curPmulR.x + (float)curVoxel.x , curPmulR.y + (float)curVoxel.y , curPmulR.z + (float)curVoxel.z };
                
                // old version (don`t rotate)
                //Point curXsumP = {curVoxel.x + p_x, curVoxel.y + p_y, curVoxel.z + p_z};
                
                //cout << curXmulR.x << " " << curXmulR.y << " " << curXmulR.z << endl;
                //cout << curXmulRsumP.x << " " << curXmulRsumP.y << " " << curXmulRsumP.z << endl;
                //cout << curVoxel.x << " " << curVoxel.y << " " << curVoxel.z << endl;
                //cout << curXsumP.x << " " << curXsumP.y << " " << curXsumP.z << endl;
                //cout << "============" << endl;
                
                if( isValidPoint(curPmulRsumX) ){
                    // old version (don`t rotate)
                    //J += getFxFromVoxel(curXsumP) * qValues[dir_i][p_x+half_l][p_y+half_l][p_z+half_l];
                    
                    // new version (rotate)
                    // J += (p*R + x) * q(d; p)
                    J += interpolateOneChanValue(curPmulRsumX, mat3dFx) * qValues[dir_i][p_x+half_l][p_y+half_l][p_z+half_l];
                    //cout << "current J is: " << J << endl;
                    //cout << "curXsumP.x: "<< curXsumP.x << ", curXsumP.y: "<< curXsumP.y << ", curXsumP.z: " << curXsumP.z <<endl;
                    //cout << dir_i << " " << p_x+half_l << " " << p_y+half_l << " " << p_z+half_l << endl;
                    }
                
            }
        }
    }
    //cout << "return J is" << J << endl;
    return J;
}

//void showJvalueDot(int x, int y, int z, int dir_i, vector<Direction> &setOfDirections, vector<pair<float, float> > &degreesByDirections);
//void showJvalueDot(int x, int y, int z, int dir_i, vector<Direction> &setOfDirections, vector<pair<float, float> > &degreesByDirections){
//    Point voxel = {x, y, z};
//    float J = calculJ(voxel, setOfDirections[dir_i], dir_i, degreesByDirections);
//    cout << "(" << x << ", " << y << ", " << z << ") dir_i: " << dir_i << "Dir: (" << setOfDirections[dir_i].x << ", " << setOfDirections[dir_i].y << ", "<< setOfDirections[dir_i].z << "), J: " << J << endl;
//}

void showAllJvaluesDot(int x, int y, int z, vector<Direction> &setOfDirections);
void showAllJvaluesDot(int x, int y, int z, vector<Direction> &setOfDirections){
    for(int i=0; i<(int)setOfDirections.size(); i++){
        //showJvalueDot(x, y, z, i, setOfDirections);
    }
}



void calculVoxelDirection(Point &curPoint, cv::Mat &mat, cv::Mat &matFx, cv::Mat &matDir, cv::Mat &matJ, vector<Direction> &setOfDirections, vector<pair<float, float> > &degreesByDirections);
void calculVoxelDirection(Point &curPoint, cv::Mat &mat, cv::Mat &matFx, cv::Mat &matDir, cv::Mat &matJ, vector<Direction> &setOfDirections, vector<pair<float, float> > &degreesByDirections){
    int rows = matFx.rows;
    int cols = matFx.cols;
    
    //for each Voxel x
    for(int i=0; i<rows; i++){ //rows
        curPoint.y = i;
        for(int j=0; j<cols; j++){ //cols
            curPoint.x = j;
            
            // if f(x) = 0
            if(matFx.at<float>(i,j) == 0){
                
                // Calculate Direction(x)
                // find maxJ
                float maxJ = -9999;
                
                //for each d in setOfDirections
                for(int dir_i=0; dir_i<(int)setOfDirections.size(); dir_i++){
                    Direction d = setOfDirections[dir_i];
                    
                    //cout << "i: "<< i << ", y: " << j << ", dir_i: "<< dir_i << "dir: (" << d.x << ", " << d.y << ", " << d.z << ")" <<endl;
                    //Calculate J(x,d)
                    float tempJ = calculJ(curPoint, d, dir_i, degreesByDirections);
                    //cout << "maxJ: " << maxJ << ", tempJ: " << tempJ << endl;
                    if(tempJ > maxJ){
                        maxJ = tempJ;
                        //matDir.r/g/b = d.x/d.y/d.z;
                        
                        //cout << "d.x: " << abs(d.x) << "d.y: "<< abs(d.y) << "d.z: "<< abs(d.z) << endl;
                        int color_r = abs(d.x) * 255;
                        int color_g = abs(d.y) * 255;
                        int color_b = abs(d.z) * 255;
                        
                        matDir.at<cv::Vec3b>(i,j)[0] = color_b; //Blue = Z
                        matDir.at<cv::Vec3b>(i,j)[1] = color_g; //Green = Y
                        matDir.at<cv::Vec3b>(i,j)[2] = color_r; //Red = X
                    }
                    
                }
                //int color_b = matDir.at<cv::Vec3b>(i,j)[0];
                //int color_g = matDir.at<cv::Vec3b>(i,j)[1];
                //int color_r = matDir.at<cv::Vec3b>(i,j)[2];
                
                //save max j value; -> don`t need -> need.... for Dynaminc Epsilon_j
                matJ.at<float>(i,j) = maxJ;
                
                //cout << "MAX J: "<< maxJ << endl;
                //cout << "calcul direction x: " << curPoint.x << ", y: " << curPoint.y << ", z: " << curPoint.z << endl;
                //cout << "R: " << color_r << ", G: " << color_g << ", B: " << color_b << endl;
                
            }
            else{
                //cout << i << j << "is f(x) == 1" << endl;
                matDir.at<cv::Vec3b>(i,j)[0] = 0;
                matDir.at<cv::Vec3b>(i,j)[1] = 0;
                matDir.at<cv::Vec3b>(i,j)[2] = 0;
            }
        }
        cout << "processing: " << i << "/" << rows << endl;
    }
}

void makeFinalCTmatrix(cv::Mat &mat, cv::Mat &matFx, cv::Mat &matJ, cv::Mat &matCT);
void makeFinalCTmatrix(cv::Mat &mat, cv::Mat &matFx, cv::Mat &matJ, cv::Mat &matCT){
    int rows = matFx.rows;
    int cols = matFx.cols;
    
    //for each Voxel x
    for(int i=0; i<rows; i++){ //rows
        for(int j=0; j<cols; j++){ //cols
            if(matFx.at<float>(i,j) == 0){
                matCT.at<float>(i,j) =  (matJ.at<float>(i,j) > EPSILON_J)? mat.at<float>(i,j) : 0;
            }
            else{
                matCT.at<float>(i,j) = 0;
            }
        }
    }
}


void debugSetToDirFromFx(cv::Mat &matFx, cv::Mat &matDir);
void debugSetToDirFromFx(cv::Mat &matFx, cv::Mat &matDir){
    int cols = matFx.cols;
    int rows = matFx.rows;
    
    for(int i=0; i<cols; i++){
        for(int j=0; j<rows; j++){
            if(matFx.at<float>(i,j) == 0 ){
                matDir.at<cv::Vec3b>(i,j)[0] = 255;
                matDir.at<cv::Vec3b>(i,j)[1] = 255;
                matDir.at<cv::Vec3b>(i,j)[2] = 0;
            }
        }
    }
}


void calculSetOfDirections(vector<Direction> &setOfDirections, vector<pair<float, float> > &degreesByDirections);
void calculSetOfDirections(vector<Direction> &setOfDirections, vector<pair<float, float> > &degreesByDirections){
    
    float dividedTheta = 2 * M_PI / N_THETA;
    float dividedZ = 2.0 / N_Z;
    
    for(int i_z=0; i_z<=N_Z; i_z+= 1){
        float z = -1 + i_z * dividedZ;
        pair<int, int> degrees;
        degrees.first = acos(z);
        for(int i_theta=0; i_theta <=N_THETA; i_theta += 1){
            float theta = i_theta * dividedTheta;
            degrees.second = theta;
            
            float x = (sqrt( 1 - z * z )) * cos(theta);
            float y = (sqrt( 1 - z * z )) * sin(theta);
            Direction tempDir = {x, y, z};
            setOfDirections.push_back(tempDir);
            degreesByDirections.push_back(degrees);
        }
    }
    num_d = (int)setOfDirections.size();
}




//for debug
void showAllDirections(vector<Direction> &setOfDirections){
    for(int i= 0; i<(int)setOfDirections.size(); i++){
        Direction t = setOfDirections[i];
        cout << "[" << i << "]" << "x: "<< t.x << "  y: "<< t.y << "  z: "<< t.z << endl;
    }
}

void showAllFx(vector<cv::Mat> &mat);
void showAllFx(vector<cv::Mat> &mat){
    for(int i= 0; i<(int)mat.size(); i++){
        cv::Mat fx = mat[i];
        
        int rows = fx.rows;
        int cols = fx.cols;
        
        //f(x)
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                cout << fx.at<float>(i,j) << endl;
            }
        }
    }
}

//for debug
void drawRowLine(cv::Mat &mat, int lineNumber);
void drawColLine(cv::Mat &mat, int lineNumber);

void drawColLine(cv::Mat &mat, int lineNumber){
    int cols = mat.cols;
    
    for(int i=0; i<cols; i++){
        mat.at<cv::Vec3b>(i,lineNumber)[0] = 0; //B
        mat.at<cv::Vec3b>(i,lineNumber)[1] = 200; //G
        mat.at<cv::Vec3b>(i,lineNumber)[2] = 200; //R
    }
}

void drawRowLine(cv::Mat &mat, int lineNumber){
    int rows = mat.rows;
    
    for(int i=0; i<rows; i++){
        mat.at<cv::Vec3b>(lineNumber, i )[0] = 0; //B
        mat.at<cv::Vec3b>(lineNumber, i )[1] = 200; //G
        mat.at<cv::Vec3b>(lineNumber, i )[2] = 200; //R
    }
}
void drawSquare(cv::Mat &mat, int botRow, int botCol, int topRow, int topCol);
void drawSquare(cv::Mat &mat, int botRow, int botCol, int topRow, int topCol){
    for(int i=botRow; i<=topRow; i++){
        for(int j=botCol; j<=topCol; j++){
            mat.at<cv::Vec3b>(i, j)[0] = 0; //B
            mat.at<cv::Vec3b>(i, j)[1] = 200; //G
            mat.at<cv::Vec3b>(i, j)[2] = 200; //R
        }
    }
    
}

void drawDot(cv::Mat &mat, int row, int col);
void drawDot(cv::Mat &mat, int row, int col){
    drawSquare(mat, row, col, row, col);
}
//============ for debug


void makeDensityFromDir(cv::Mat &matDir, cv::Mat &matDensity);
void makeDensityFromDir(cv::Mat &matDir, cv::Mat &matDensity){
    
    int rows = matDir.rows;
    int cols = matDir.cols;
    
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            float density = 0;
            float color_b = (matDir.at<cv::Vec3b>(i,j)[0])/255;
            float color_g = (matDir.at<cv::Vec3b>(i,j)[1])/255;
            float color_r = (matDir.at<cv::Vec3b>(i,j)[2])/255;
            
            density += (color_b * color_b);
            density += (color_r * color_r);
            density += (color_g * color_g);
            density = sqrt(density);
            
            matDensity.at<float>(i,j) = density;
        }
    }
}

void showDotRGBvalue(cv::Mat &mat, int row, int col);
void showDotRGBvalue(cv::Mat &mat, int row, int col){
    int i = row;
    int j = col;
    cout << "(" << row << ", " << col << ")";
    cout << " B: "<< (int)mat.at<cv::Vec3b>(i,j)[0] << ",G: "<< (int)mat.at<cv::Vec3b>(i,j)[1] << ",R: "<< (int)mat.at<cv::Vec3b>(i,j)[2] << endl;
}

void showSquareRGBvalue(cv::Mat &mat, int botRow, int botCol, int topRow, int topCol);
void showSquareRGBvalue(cv::Mat &mat, int botRow, int botCol, int topRow, int topCol){
    for(int i=botRow; i<topRow; i++){
        for(int j=botCol; j<topCol; j++){
            showDotRGBvalue(mat, j, i);
        }
    }
}

void showDotSingleValue(cv::Mat &mat, int row, int col);
void showDotSingleValue(cv::Mat &mat, int row, int col){
    int i = row;
    int j = col;
    cout << "(" << row << ", " << col << ")";
    cout << " Value: " << mat.at<float>(i,j) << endl;
}

void makeBlobMat(cv::Mat &im, cv::Mat &im_with_keypoints){
    // Setup SimpleBlobDetector parameters.
    cv::SimpleBlobDetector::Params params;

    // Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 200;

    // Filter by Area.
    params.filterByArea = true;
    params.minArea = 5;
    //params.maxArea = 1500;

    // Filter by Circularity
    params.filterByCircularity = true;
    params.minCircularity = 0.1;

    // Filter by Convexity
    params.filterByConvexity = true;
    params.minConvexity = 0.87;

    // Filter by Inertia
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;


    // Storage for blobs
    vector<cv::KeyPoint> keypoints;


#if CV_MAJOR_VERSION < 3   // If you are using OpenCV 2

    // Set up detector with params
    cv::SimpleBlobDetector detector(params);

    // Detect blobs
    detector.detect( im, keypoints);
#else 

    // Set up detector with params
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);   

    // Detect blobs
    detector->detect( im, keypoints);
#endif 

    // Draw detected blobs as red circles.
    // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
    // the size of the circle corresponds to the size of blob

    drawKeypoints( im, keypoints, im_with_keypoints, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

    //print center of each blobs
    cout << "Number of blob: " << (int)keypoints.size() << endl;
    for(int i=0; i<(int)keypoints.size(); i++){
        cout << i << " center: (i:" << keypoints[i].pt.x << ", j: " << keypoints[i].pt.y << ")"<<endl;
    }
}

void makeIx(cv::Mat &matSubX, cv::Mat &matIx);
void makeIx(cv::Mat &matSubX, cv::Mat &matIx){

    for(int iy=0; iy<sy; iy++){ // y
        for(int iz=0; iz<sz; iz++){ // z
            float dirx = matSubX.at<cv::Vec3b>(iz,iy)[0];

            //cout << "makeIx(): " << dirx/255 << endl;
            if(dirx/255 > EPSILON_I){
                matIx.at<unsigned char>(iz,iy) = (unsigned char)0; //y, z. 1 by paper
            }
            else{
                matIx.at<unsigned char>(iz,iy) = (unsigned char)255; //y, z. 0 by paper
            }
        }
    }
}

void changeEpsilonD(int pos, void *param)
{
    cv::Mat &mat = *(cv::Mat*)param;
    cv::Mat matFx = cv::Mat(mat.rows, mat.cols, CV_32F, float(0));

    EPSILON_D = (float)pos/100;
    
    calculFx(mat, matFx);
    cout << "current EPSILON_D is " << EPSILON_D << endl;
    imshow("Hello OpenCV", matFx);
}

void changeEpsilonJ(int pos, void *param){
    cv::Mat **mats = (cv::Mat**)malloc(sizeof(cv::Mat*) * 3);
    
    mats = (cv::Mat**)param;
    cv::Mat &mat = *(mats[0]);
    cv::Mat &matFx = *(mats[1]);
    cv::Mat &matJ = *(mats[2]);
    
    cv::Mat matCT = cv::Mat(mat.rows, mat.cols, CV_32F, float(0));
    
    EPSILON_J = -(float)pos/5;
    makeFinalCTmatrix(mat, matFx, matJ, matCT);
    cout << "current EPSILON_J is " << EPSILON_J << endl;
    imshow("Hello OpenCV", matCT);
}

void changeEpsilonI(int pos, void *param){
    cv::Mat &matSubX = *(cv::Mat*)param;
    cv::Mat im_with_keypoints;
    cv::Mat matIx = cv::Mat(sz, sy, CV_8UC1);
    makeIx(matSubX, matIx);
    cout << "current EPSILON_I is " << EPSILON_I << endl;
    EPSILON_I = (float)pos/10;
    makeBlobMat(matIx, im_with_keypoints);

    imshow("Hello OpenCV", im_with_keypoints );
}

float calcul3dVectorAbs(Direction &direction){
    float result = 0;
    direction.x /= 255;
    direction.y /= 255;
    direction.z /= 255; 

    result += (direction.x * direction.x);
    result += (direction.y * direction.y);
    result += (direction.z * direction.z);
    return sqrt(result);
}

void divideSubVolumeX(vector<cv::Mat> &matDir3d, vector<cv::Mat> &matSubX3d);
void divideSubVolumeX(vector<cv::Mat> &matDir3d, vector<cv::Mat> &matSubX3d){
    
    //cout << "sx: " << sx << ", sy: " << sy << " sz: " << sz << endl;
    for(int ix=0; ix<sx; ix++){ // x
        cv::Mat planeYZ = cv::Mat(sz, sy, CV_8UC3); // y(rows), z(planes)
        matSubX3d.push_back(planeYZ);

        for(int iy=0; iy<sy; iy++){ // y
            for(int iz=0; iz<sz; iz++){ // z
                unsigned char dirx = (int)matDir3d[iz].at<cv::Vec3f>(iy,ix)[0];
                unsigned char diry = (int)matDir3d[iz].at<cv::Vec3f>(iy,ix)[1];
                unsigned char dirz = (int)matDir3d[iz].at<cv::Vec3f>(iy,ix)[2];

                // dirx is biggest
                if(diry > dirx || dirz > dirx){
                    matSubX3d[ix].at<cv::Vec3b>(iz,iy)[0] = 0;
                    matSubX3d[ix].at<cv::Vec3b>(iz,iy)[1] = 0;
                    matSubX3d[ix].at<cv::Vec3b>(iz,iy)[2] = 0;
                } 
                else{
                    matSubX3d[ix].at<cv::Vec3b>(iz,iy)[0] =  dirx;
                    matSubX3d[ix].at<cv::Vec3b>(iz,iy)[1] =  diry;
                    matSubX3d[ix].at<cv::Vec3b>(iz,iy)[2] =  dirz;
                }
            }
        }
        cout << "dividing.. " << ix << "/" << sx << endl;
    }
}

int main(int argc, const char * argv[]){

    char inputType;
    int inputPlane = -1;

    // input valid check
    if(argc != 2 && argc != 3){
        
        fprintf(stderr,"<TYPE> \n");
        fprintf(stderr,"o: Original CT Image\n");
        fprintf(stderr,"b: Binary CT Image\n");
        fprintf(stderr,"s: skip Original, Binary CT Images\n");
        fprintf(stderr,"Direction RGB Image, Cleaned CT Image are always printed\n");
        fprintf(stderr,"<IMAGE_NUMBER> \n");
        fprintf(stderr,"RGB, Cleand Image Number. (int)0 ~ Size of Z\n");
        fprintf(stderr,"<Usage> \n");
        fprintf(stderr,"./main TYPE IMAGE_NUMBER (ex: ./main b 100)\n");
        return -1;
    }
    inputType = argv[1][0];
    if(inputType != 'o' && inputType != 'b' && inputType != 's'){
        fprintf(stderr,"<TYPE>\nmust be 'o', 'b', 's'\n");
        fprintf(stderr,"<Usage> \n");
        fprintf(stderr,"./main TYPE IMAGE_NUMBER (ex: ./main b 100)\n");
        return -1;
    }

    //start processing timer
    clock_t begin = clock(); 



    // Step​ ​0.​ ​Reading​ ​3D-CT​ ​images

    //////////////////////////////////////////////////////// read VOL data
    FILE    *fp_sour;
    unsigned char buff[48]; //48byte
    size_t   n_size;

    fp_sour = fopen("./Pramook_black_velvet_3.03um_80kV_down.vol", "rb");
    n_size = fread(buff, 1, 48, fp_sour);
    readHeader(buff);

    // input valid check
    if(argc == 3){
        inputPlane = atoi(argv[2]);

        if(sz <= inputPlane){
            fprintf(stderr, "PLANE_NUMBER must be 0 ~ %d\n", sz-1 ); 
            return -1;
        }
    }

    vector<float> data((((long long)sx)*sy)*sz);
    readData(data, fp_sour, 1);

    fclose(fp_sour);
    ///////////////////////////////////////////////////////

    string windowName = "Hello OpenCV";
    cv::Mat mat, matFx, matDir, matDensity ,matCT, matJ;

    vector<cv::Mat> mat3d, mat3dDir;
    vector<Direction> setOfDirections;
    vector<pair<float, float> > degreesByDirections;

    cout << "Start processing" << endl;



    // Step​ ​1.​ ​Determining​ ​Epsilon​ ​D

    for(int curFileNum = 0; curFileNum < sz; curFileNum++){
        // read input image from mat
        readNextInput(curFileNum, mat, data);

        // Print Original CT Image
        if(inputType == 'o'){
            if(inputPlane == -1){
                printf("Current Original CT Image Number: %d \n", curFileNum);
                cv::imshow(windowName, mat);
                cv::waitKey(0);
            }
        }
        mat3d.push_back(mat);

        // caculate f(x)
        matFx = cv::Mat(mat.rows, mat.cols, CV_32F, float(0));
        calculFx(mat, matFx);
        mat3dFx.push_back(matFx);

        // Print Binary CT Image
        if(inputType == 'b'){
            if(inputPlane == -1){
                printf("Current Binary CT Image Number: %d \n", curFileNum);
                cv::imshow(windowName, matFx);
                cv::createTrackbar("threahold D", windowName, &EPSILON_D_BASE, 100, changeEpsilonD, (void*)&mat);
                cv::setTrackbarPos("threahold D", windowName, 55);
                cv::waitKey(0);
            }
        }
    }
    // showAllFx(mat3dFx); // debug



    // input valid check
    if(argc == 2){
        fprintf(stderr, "<Program End>\n");
        fprintf(stderr, "If you print Direction(RGB) Image & Clenaed CT Image\n");
        fprintf(stderr, "You should input IMAGE_NUMBER\n"); 
        fprintf(stderr,"<Usage> \n");
        fprintf(stderr,"./main TYPE IMAGE_NUMBER (ex: ./main b 100)\n");
        return -1;
    }


    // Step​ ​2.​ ​Computing​ ​fiber​ ​direction​ ​Set
    // (It takes a long time in this Step)

    // 1) Getting set of directions
    calculSetOfDirections(setOfDirections, degreesByDirections);
    //showAllDirections(degreesByDirections); //debug

    calculAllQs(setOfDirections);
    ROWS = mat.rows;
    COLS = mat.cols;

    int curFileNum = inputPlane;
    //    for(int curFileNum = 0; curFileNum < sz; curFileNum++){

    mat = mat3d[curFileNum];
    matFx = mat3dFx[curFileNum];
    cout << "curFileNum: " << curFileNum << endl;

    // make matirx of direction
    Point curPoint = {0, 0, curFileNum};
    matDir = cv::Mat(mat.rows, mat.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    // make matrix of Final CT values
    matCT = cv::Mat(matFx.rows, matFx.cols, CV_32F, float(0));
    // make matrix of Maximum J value of each voxel
    matJ = cv::Mat(matFx.rows, matFx.cols, CV_32F, float(0));

    // 2) Finding the J value
    // 3) Determining the direction
    calculVoxelDirection(curPoint, mat, matFx, matDir, matJ, setOfDirections, degreesByDirections);
    makeFinalCTmatrix(mat, matFx, matJ, matCT);

    mat3dDir.push_back(matDir);
    mat3dCT.push_back(matCT);
    mat3dJ.push_back(matJ);



    // Density is needed?
    // matDensity = cv::Mat(mat.rows,mat.cols, CV_32F, float(0));
    // makeDensityFromDir(matDir, matDensity);



    //end processing timer
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    cout << "Processing time: " << time_spent << "s" << endl;

    // Print Direction RGB Image
    cv::imshow(windowName, matDir);
    cv::waitKey(0);
    


    // Step 3. Determining Epsilon J
    cv::Mat *mats[3] = {&mat, &matFx, &matJ};
    // Print Cleaned CT Image
    cv::imshow(windowName, matCT);
    cv::createTrackbar("threahold J", windowName, &EPSILON_J_BASE, 100, changeEpsilonJ, (void*)mats);
    cv::setTrackbarPos("threahold J", windowName, 5);
    cv::waitKey(0);
    
    //    }
    // >>>>> The others Processing(Step4~6) are implemented by HJ<<<<<
    // Github: 

    /////////////////////////////////////////////// read_VOL_CH3
    // FILE    *fp_sour2;
    // unsigned char buff2[48]; //48byte
    // size_t   n_size2;

    // fp_sour2 = fopen("./3-2_dir_down.vol", "rb");
    // n_size2 = fread(buff2, 1, 48, fp_sour2);

    // readHeader(buff2);

    // vector<float> data3CH(((((long long)sx)*sy)*sz)*channels);
    // readData(data3CH, fp_sour2, channels);
    // //printData(data3CH);

    // fclose(fp_sour2);
    //////////////////////////////////////////////

    // vector<cv::Mat> matDir3d  from HN
    // vector<cv::Mat> matDir3d;
    // for (int curFileNum = 0; curFileNum < sz; curFileNum++) {
    //     //  //read input image
    //     cv::Mat matCH3;
    //     readNextInputCH3(curFileNum, matCH3, data3CH);
    //     //imshow("matCH3 image", matCH3);
    //     //cv::waitKey(0);
    //     matDir3d.push_back(matCH3);
    // }
    //////////////////////////////////////////////
    
    // divide to 3sub Volumes
    // vector<cv::Mat> matIx3d, matSubX3d;
    // divideSubVolumeX(matDir3d, matSubX3d); // (dir -> subX)

    // for(int i=0; i<(int)matSubX3d.size(); i++){

    //     cv::Mat IxplaneYZ = cv::Mat(sz, sy, CV_8UC1); // y(rows), z(planes)
        
    //     makeIx(matSubX3d[i], IxplaneYZ); // (subX -> Ix)
    //     matIx3d.push_back(IxplaneYZ);
    // }
    
    // for(int i=0; i<(int)matIx3d.size(); i++){
    //     cv::Mat im_with_keypoints;
    //     cout << "current Image: X voxel " << i << endl;
    //     makeBlobMat(matIx3d[i], im_with_keypoints); // (Ix -> blobs)
        
    //     // Show blobs
    //     imshow(windowName, im_with_keypoints );
    //     cv::createTrackbar("threahold I", windowName, &EPSILON_I_BASE, 10, changeEpsilonI, (void*)&matSubX3d[i]);
    //     cv::waitKey(0);
    // }    
    return 0;
}