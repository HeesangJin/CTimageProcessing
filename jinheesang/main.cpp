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
#include <ctime>

#include "opencv2/opencv.hpp"

#define NUM_PLANES 150
#define EPSILON_D 0.55
#define EPSILON_J -1

#define N_THETA 10
#define N_Z 20
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


vector<cv::Mat> mat3dFx, mat3dCT;

int ROWS, COLS;
int num_d;

//vector[num_d][LENGTH_L+1]
vector<vector<vector<vector<float> > > > qValues;


string type2str(int type);
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
    input = cv::imread(filename, cv::IMREAD_UNCHANGED);
    input.convertTo(mat, CV_8U, 1.0/255.0);
    mat.convertTo(mat, CV_32F, 1.0/255.0);
    
//    for(int i=500; i<mat.rows; i++){
//        for(int j=500; j<mat.cols; j++){
//            cout << mat.at<float>(i,j) << endl;
//        }
//    }
    
    
    cout << "Image dimensions = " << mat.size() << endl;
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


bool isValidPoint(Point point);
bool isValidPoint(Point point){
    if(point.x < 0 || point.x >= COLS || point.y < 0 || point.y >= ROWS || point.z < 0 || point.z >= NUM_PLANES)
        return false;
    return true;
}


float getFxFromVoxel(Point voxel);
float getFxFromVoxel(Point voxel){
    return (mat3dFx[voxel.z]).at<float>(voxel.y, voxel.x);
}


float calculJ(Point &curVoxel, Direction &d, int dir_i);
float calculJ(Point &curVoxel, Direction &d, int dir_i){
    float J=0;
    
    //each p in V(V`s size is l)
    int half_l = LENGTH_L/2;
    for(int p_x= -half_l; p_x<=half_l; p_x++){
        for(int p_y= -half_l; p_y<=half_l; p_y++){
            for(int p_z= -half_l; p_z<=half_l; p_z++){
                Point curP = {p_x, p_y, p_z};
                Point curXsumP = {p_x + curVoxel.x , p_y + curVoxel.y , p_z + curVoxel.z };
                
                //cout << curXsumP.x << " " << curXsumP.y << " " << curXsumP.z << endl;
                // J += f(x+p) * q(d;p)
                if( isValidPoint(curXsumP) ){
                    //cout << "curXsumP.x: "<< curXsumP.x << ", curXsumP.y: "<< curXsumP.y << ", curXsumP.z: " << curXsumP.z <<endl;
                    
                    //cout << dir_i << " " << p_x+half_l << " " << p_y+half_l << " " << p_z+half_l << endl;
                    J += getFxFromVoxel(curXsumP) * qValues[dir_i][p_x+half_l][p_y+half_l][p_z+half_l];
                    //cout << "current J is: " << J << endl;
                }
            }
        }
    }
    //cout << "return J is" << J << endl;
    return J;
}

void showJvalueDot(int x, int y, int z, int dir_i, vector<Direction> &setOfDirections);
void showJvalueDot(int x, int y, int z, int dir_i, vector<Direction> &setOfDirections){
    Point voxel = {x, y, z};
    float J = calculJ(voxel, setOfDirections[dir_i], dir_i);
    cout << "(" << x << ", " << y << ", " << z << ") dir_i: " << dir_i << "Dir: (" << setOfDirections[dir_i].x << ", " << setOfDirections[dir_i].y << ", "<< setOfDirections[dir_i].z << "), J: " << J << endl;
}

void showAllJvaluesDot(int x, int y, int z, vector<Direction> &setOfDirections);
void showAllJvaluesDot(int x, int y, int z, vector<Direction> &setOfDirections){
    for(int i=0; i<(int)setOfDirections.size(); i++){
        showJvalueDot(x, y, z, i, setOfDirections);
    }
}


void calculVoxelDirection(Point &curPoint, cv::Mat &mat, cv::Mat &matFx, cv::Mat &matDir, cv::Mat &matCT, vector<Direction> &setOfDirections);
void calculVoxelDirection(Point &curPoint, cv::Mat &mat, cv::Mat &matFx, cv::Mat &matDir, cv::Mat &matCT, vector<Direction> &setOfDirections){
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
                    float tempJ = calculJ(curPoint, d, dir_i);
                    //cout << "maxJ: " << maxJ << ", tempJ: " << tempJ << endl;
                    if(tempJ > maxJ){
                        maxJ = tempJ;
                        //matDir.r/g/b = d.x/d.y/d.z;
                        
                        //cout << "d.x: " << abs(d.x) << "d.y: "<< abs(d.y) << "d.z: "<< abs(d.z) << endl;
                        int color_r = abs(d.x) * 255;
                        int color_g = abs(d.y) * 255;
                        int color_b = abs(d.z) * 255;
                        
                        matDir.at<cv::Vec3b>(i,j)[0] = color_b;
                        matDir.at<cv::Vec3b>(i,j)[1] = color_g;
                        matDir.at<cv::Vec3b>(i,j)[2] = color_r;
                    }
                    
                }
                int color_b = matDir.at<cv::Vec3b>(i,j)[0];
                int color_g = matDir.at<cv::Vec3b>(i,j)[1];
                int color_r = matDir.at<cv::Vec3b>(i,j)[2];
                
                //save max j value; -> don`t need
                //matMaxJvalues.at<float>(i,j) = maxJ;
                
                matCT.at<float>(i,j) =  (maxJ > EPSILON_J)? mat.at<float>(i,j) : 0;
                
                //cout << "MAX J: "<< maxJ << endl;
                //cout << "calcul direction x: " << curPoint.x << ", y: " << curPoint.y << ", z: " << curPoint.z << endl;
                //cout << "R: " << color_r << ", G: " << color_g << ", B: " << color_b << endl;
                
            }
            else{
                //cout << i << j << "is f(x) == 1" << endl;
                matDir.at<cv::Vec3b>(i,j)[0] = 0;
                matDir.at<cv::Vec3b>(i,j)[1] = 0;
                matDir.at<cv::Vec3b>(i,j)[2] = 0;
                
                matCT.at<float>(i,j) = 0;
            }
        }
        cout << "processing: " << i << "/" << rows << endl;
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


void calculSetOfDirections(vector<Direction> &setOfDirections);
void calculSetOfDirections(vector<Direction> &setOfDirections){
    
    float dividedTheta = 2 * M_PI / N_THETA;
    float dividedZ = 2.0 / N_Z;
    
    for(int i_z=0; i_z<=N_Z; i_z+= 1){
        float z = -1 + i_z * dividedZ;
        cout << dividedZ << endl;
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

int main(int argc, const char * argv[]){
    clock_t begin = clock();
    
    string windowName = "Hello OpenCV";
    cv::Mat mat, matFx, matDir, matDensity ,matCT;
    
    
    vector<cv::Mat> mat3d, mat3dDir;
    vector<Direction> setOfDirections;
    
    
    cout << "Start processing" << endl;
    
    
    calculSetOfDirections(setOfDirections);
    
    //debug
    showAllDirections(setOfDirections);
    
    
    for(int curFileNum = 0; curFileNum < NUM_PLANES; curFileNum++){
        //read input image
        readNextInput(curFileNum, mat);
        
        mat3d.push_back(mat);
        //ok
        
        //caculate f(x)
        matFx = cv::Mat(mat.rows, mat.cols, CV_32F, float(0));
        calculFx(mat, matFx);
        mat3dFx.push_back(matFx);
        //ok
        
        //DEBUG
        //        cv::Mat matTest;
        //        matTest = cv::Mat(matFx.rows, matFx.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        //        debugToRgb(matFx, matTest);
        //
        //cv::imshow(windowName, mat);
        //cv::waitKey(0);
    }
    //showAllFx(mat3dFx);
    
    calculAllQs(setOfDirections);
    ROWS = mat.rows;
    COLS = mat.cols;
    
    int curFileNum = 100;
    //    for(int curFileNum = 0; curFileNum < NUM_PLANES; curFileNum++){
    
    mat = mat3d[curFileNum];
    matFx = mat3dFx[curFileNum];
    cout << "curFileNum: " << curFileNum << endl;
    
    // make matirx of direction
    Point curPoint = {0, 0, curFileNum};
    matDir = cv::Mat(mat.rows, mat.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // make matrix of Final CT values
    matCT = cv::Mat(matFx.rows,matFx.cols, CV_32F, float(0));
    
    // Calculating Directions
    calculVoxelDirection(curPoint, mat, matFx, matDir, matCT, setOfDirections);
    
    mat3dDir.push_back(matDir);
    mat3dCT.push_back(matCT);
    
    //matDensity = cv::Mat(mat.rows,mat.cols, CV_32F, float(0));
    //makeDensityFromDir(matDir, matDensity);
    
    /* debug - draw lines in yellow */
    //drawRowLine(matDir, 100);
//    drawColLine(matDir, 330); //left
//    drawColLine(matDir, 370); //right
//    drawRowLine(matDir, 585); //top
//    drawRowLine(matDir, 625); //bottom
//    //top-left is (0,0)
//    
//    drawDot(matDir, 350, 600);
//    drawDot(matDir, 351, 600);
//    drawDot(matDir, 352, 600);
//    drawDot(matDir, 353, 600);
//    drawDot(matDir, 354, 600);
//    drawDot(matDir, 355, 600);
//    drawDot(matDir, 356, 600);
//    drawDot(matDir, 357, 600);
//    drawDot(matDir, 358, 600);
//    drawDot(matDir, 359, 600);
//    drawDot(matDir, 360, 600);
//    
//    
//    //debug
//    showAllJvaluesDot(350, 600, curFileNum, setOfDirections);
    
    //drawColLine(matDir, 400);
    //drawSquare(matDir, 320, 580, 380, 630); // bottom (x,y) and top (x,y)
    
    //showSquareRGBvalue(matDir, 330, 585, 370, 625); // bottom (x,y) and top (x,y)
    
    //showDotRGBvalue(matDir, 350, 605); // not showDotRGBvalue(matDir, 605, 350);
    
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    cout << "Processing time: " << time_spent << "s" << endl;
    
    cv::imshow(windowName, matDir);
    cv::waitKey(0);
    
    cv::imshow(windowName, matCT);
    cv::waitKey(0);
    
    //    }
    
    return 0;
}

