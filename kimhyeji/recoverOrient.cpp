#define _USE_MATH_DEFINES

#include "opencv2/opencv.hpp"
//#include "GL/glew.h"
//#include "GL/glut.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#define ROW 250
#define COL 250 //988
#define IMG_NUM 90

using namespace cv;
using namespace std;

typedef pair<int, pair<int, int>> vec3D;
typedef pair<float, pair<float, float>> direction;
typedef vector < direction > directionSet;

int h = 12;
int s = 3;
int t = 4;
int dirNum = 10;
int h_half = h / 2;

string fileLocate = "C:\\Users\\kimhyeji\\Desktop\\UCI\\project\\Pramook_black_velvet_3.03um_80kV_TIFF\\";

/*
test code for deciding threshold
*/
void testThreshold(const Mat testImg, int from, int to) {
	Mat binary_img;
	for (int i = from; i < to; i++) {
		threshold(testImg, binary_img, i, 255, THRESH_BINARY);
		imshow("Display window", binary_img);
		printf("%d", i);
		waitKey(0);
	}
}

/*
open image files and store to 3D volume
*/
void makeVolume(vector<Mat> &v) {
#pragma omp parallel for schedule(dynamic,8)
	for (int i = 1; i <= IMG_NUM; i++) {
		Mat img;
		string image = fileLocate + format("%04d.tiff", i);
		img = imread(image, CV_LOAD_IMAGE_GRAYSCALE);
		normalize(img, img, 0.0, 1, CV_MINMAX, CV_32F);
		threshold(img, img, 0.4, 1, THRESH_BINARY_INV);
		img.convertTo(img, CV_8U, 1,0);
		v[i - 1] = img;
		img.release();
	}
}

/*
make set of direction vectors
*/
void makedirectionSet(directionSet &v) {
	float M_PI2 = M_PI * 2;
	float intervalTheta = M_PI2 / (dirNum - 1);
	float intervalZ = 2.0 / (dirNum - 1);
	int count = 0;
	for (float z = -1.0; z <= 1.0; z += intervalZ) {
		for (float theta = 0; theta <= M_PI2; theta += intervalTheta){
			float x = sqrt(1.0 - pow(z, 2)) * cos(theta);
			float y = sqrt(1.0 - pow(z, 2)) * sin(theta);
			v[count++] = {z,{ x,y }};
		}
	}
}

/*
get distance with direction vector and location p
*/
direction operator -(const vec3D &a, const direction &b) {
	return{ a.first - b.first ,{ a.second.first - b.second.first, a.second.second - b.second.second } };
}
direction operator *(float a, const direction &b) {
	return{ a * b.first ,{ a * b.second.first, a * b.second.second } };
}
float getDot(const vec3D &a, const direction &b) {
	return a.first * b.first + a.second.first * b.second.first + a.second.second * b.second.second;
}
float getDistance(const direction &direct, const vec3D &x) {
	direction dist = x - getDot(x, direct) * direct;
	return sqrt(pow(dist.first, 2) + pow(dist.second.first, 2) + pow(dist.second.second, 2));
}

/*
get Maximum convolution value
*/
direction getMaxConvolution(const directionSet &d, const vector<Mat> &volume, const vec3D &v) {
	int x = v.second.first;
	int y = v.second.second;
	int z = v.first;
	float maxValue = -99999;
	direction result;

#pragma omp parallel for schedule(dynamic,8)
	for (int i = 0; i < 100; i++) {		
		float j = 0;
		for (int p_z = 0; p_z <= h; p_z++) {
			if (z + p_z > IMG_NUM) continue;
			for (int p_x = 0; p_x <= h; p_x++) {
				if (x + p_x > ROW) continue;
				for (int p_y = 0; p_y <= h; p_y++) {
					if (y + p_y > COL) continue;
					
					float dist = getDistance(d[i], { p_z,{ p_x,p_y } });
					if (i > 99) cout << p_x << ' ' << p_y << ' ' << p_z << endl;

					float q_func = -2 * exp(-s * pow(dist, 2)) + exp(-t * pow(dist, 2));
					j += q_func;
					
				}
			}	
		}
		if (maxValue < j) {
			maxValue = j;
			result = d[i];
		}
	}
	return result;
}



int main(int argc, char **argv) {

	vector<Mat> volumeData(IMG_NUM);
	makeVolume(volumeData);

	/*
	memory problem.

	ifstream read;
	string volumeFile = "volumeData.xml";
	read.open(volumeFile);
	if (read) {
	FileStorage fs_r(volumeFile, FileStorage::READ);
	fs_r["volumeData"] >> volumeData;
	cout << "We open it" << endl;
	}
	else {
	FileStorage fs_w(volumeFile, FileStorage::WRITE);
	makeVolume(volumeData);
	fs_w << "volumeData" << volumeData;
	fs_w.release();
	}
	*/

	cout << endl << "make volume" << endl;

	directionSet dicVec(dirNum * dirNum);
	makedirectionSet(dicVec);

	cout << endl << "make directionSet" << endl;


	/*
	recover the orientation field
	*/

	//vector< Mat> orientData(IMG_NUM, Mat (ROW,COL, CV_8UC1));
	for (int z = 0; z < IMG_NUM; z++) {

		cout << z << endl;
		Mat image_dir(ROW, COL, CV_32FC3);
		Mat image_den(ROW, COL, CV_32FC1);


		for (int x = 0; x < ROW; x++) {
			cout << "x" << x << endl;
			for (int y = 0; y < COL; y++) {
				if (int(volumeData[z].at<uchar>(x, y))) {
					image_dir.at<Vec3f>(x, y)[0] = 0.0;
					image_dir.at<Vec3f>(x, y)[1] = 0.0;
					image_dir.at<Vec3f>(x, y)[2] = 0.0;
					image_den.at<float>(x, y) = 0;
				}
				else {
					direction result = getMaxConvolution(dicVec, volumeData, { z,{ x,y } });
					image_dir.at<Vec3f>(x, y)[0] = abs(result.second.first);
					image_dir.at<Vec3f>(x, y)[1] = abs(result.second.second);
					image_dir.at<Vec3f>(x, y)[2] = abs(result.first);
					float density =  sqrt(pow(result.second.first, 2) + pow(result.second.second, 2) + pow(result.first, 2));
					image_den.at<float>(x, y) = density;
					

				}
			}
		}
		imshow("dis", image_dir);
		imwrite("direction" + format("%d.jpg", z), image_dir);
		imwrite("density" + format("%d.jpg", z), image_den);
		image_dir.release();
		image_den.release();
	}
		
}



/*

glutInit(&argc, argv);
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
glutInitWindowSize(500, 500);
glutCreateWindow("test");

glClearColor(1.0, 1.0, 1.0, 0.0);
glClear(GL_COLOR_BUFFER_BIT);

glMatrixMode(GL_PROJECTION);
glLoadIdentity();

glMatrixMode(GL_MODELVIEW);
glLoadIdentity();

glColor3f(0.3, 0.3, 0.3);
glPointSize(5.0f);
glBegin(GL_POINTS);
for (int i = 0; i < 400; i++) {
float x = dicVec[i].first;
float y = dicVec[i].second.first;
float z = dicVec[i].second.second;
glVertex3f(x, y, z);
}

glEnd();
glFlush();
waitKey();
*/