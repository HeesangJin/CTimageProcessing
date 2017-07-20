#define _USE_MATH_DEFINES

#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#define ROW 1010 //1010
#define COL 988 //988
#define IMG_NUM 200 //990
#define H 13
#define DIRNUM 10
#define DIRNUMSQU 100
using namespace cv;
using namespace std;

typedef vector < Vec3f > directionSet;

int h = 12;
int s = 3;
int t = 4;
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
		threshold(img, img, 0.68, 1, THRESH_BINARY_INV);
		img.convertTo(img, CV_8U, 255, 0);
		imshow("img", img);
		waitKey(0);
		v[i - 1] = img;
		img.release();
	}
}

/*

make set of direction vectors
*/
void makedirectionSet(directionSet &v) {
	float M_PI2 = M_PI * 2;
	float intervalTheta = M_PI2 / (DIRNUM - 1);
	float intervalZ = 2.0 / (DIRNUM - 1);
	int count = 0;
	for (float z = -1.0; z <= 1.0; z += intervalZ) {
		for (float theta = 0; theta <= M_PI2; theta += intervalTheta) {
			float x = sqrt(1.0 - pow(z, 2)) * cos(theta);
			float y = sqrt(1.0 - pow(z, 2)) * sin(theta);
			v[count++] = Vec3f(z, x, y);
		}
	}
}


/*
get distance with direction vector and location p
*/
float getDistance(const Vec3f &direct, const Vec3f &x) {
	Vec3f dist = x - (x.dot(direct) * direct);
	return sqrt(pow(dist[0], 2) + pow(dist[1], 2) + pow(dist[2], 2));
}


/*
save q function values
*/
void saveDistance(float(&saveData)[DIRNUMSQU][H][H][H], const directionSet &d) {
	for (int dir = 0; dir < DIRNUMSQU; dir++) {

		for (int p_z = 0; p_z <= h; p_z++) {
			for (int p_x = 0; p_x <= h; p_x++) {
				for (int p_y = 0; p_y <= h; p_y++) {
					float dist = getDistance(d[dir], Vec3f(p_z, p_x, p_y));
					float pow_dist = dist * dist;
					float q_func = -2 * exp(-s * pow_dist) + exp(-t * pow_dist);
					saveData[dir][p_z][p_x][p_y] = q_func;
				}
			}
		}
	}
}

/*
get Maximum convolution value
*/
Vec3f getMaxConvolution(const directionSet &d, const vector<Mat> &volume, const Vec3i &v, const float(&q)[DIRNUMSQU][H][H][H]) {
	int x = v[1];
	int y = v[2];
	int z = v[0];
	float maxValue = -99999;
	Vec3f result;
	for (int i = 0; i < DIRNUMSQU; i++) {
		float j = 0;
		for (int p_z = 0; p_z <= h; p_z++) {
			if (z + p_z >= IMG_NUM) continue;
			for (int p_x = 0; p_x <= h; p_x++) {
				if (x + p_x >= ROW) continue;
				for (int p_y = 0; p_y <= h; p_y++) {
					if (y + p_y >= COL) continue;

					if (volume[z + p_z].at<uchar>(x + p_x, y + p_y) > 0)
					{
						float q_func = q[i][p_z][p_x][p_y];
						j += q_func;
					}
				}
			}
		}
		if (maxValue < j) {
			maxValue = j;
			result = d[i];
		}
	}
	if (maxValue < -0.01) {
		result = Vec3f(0, 0, 0);
	}
	return result;
}



int main(int argc, char **argv) {

	vector<Mat> volumeData(IMG_NUM);
	float qfuncData[DIRNUMSQU][H][H][H];
	makeVolume(volumeData);

	directionSet dicVec(DIRNUMSQU);
	makedirectionSet(dicVec);

	saveDistance(qfuncData, dicVec);

	/*
	recover the orientation field
	*/
	for (int z = 100; z < IMG_NUM; z++) {
		cout << z << endl;
		Mat image_dir(ROW, COL, CV_32FC3);
		Mat image_den(ROW, COL, CV_32FC1);

#pragma omp parallel for schedule(dynamic,8)
		for (int x = 0; x < ROW; x++) {
			for (int y = 0; y < COL; y++) {
				if (volumeData[z].at<uchar>(x, y) > 0) {
					image_dir.at<Vec3f>(x, y)[0] = 0;
					image_dir.at<Vec3f>(x, y)[1] = 0;
					image_dir.at<Vec3f>(x, y)[2] = 0;
					continue;
				}

				Vec3f result = getMaxConvolution(dicVec, volumeData, Vec3i(z, x, y), qfuncData);
				image_dir.at<Vec3f>(x, y)[0] = abs(result[1]) * 255;
				image_dir.at<Vec3f>(x, y)[1] = abs(result[2]) * 255;
				image_dir.at<Vec3f>(x, y)[2] = abs(result[0]) * 255;

			}

			if (x % 50 == 0) {
				cout << x << endl;
				imwrite("d" + format("%d.jpg", z), image_dir);
			}

		}
		imwrite("direction" + format("%d.jpg", z), image_dir);
		image_dir.release();
	}

}


