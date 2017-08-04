#define _USE_MATH_DEFINES

#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#define ROW 1010//1010
#define COL 988 //988
#define IMG_NUM 50 //990
#define H 13 // H + 1
#define DIRNUM 12
#define DIRNUMSQU 122 // DIRNUM * DIRNUM - (2 * DIRNUM-1)
using namespace cv;
using namespace std;

typedef vector < Vec3f > directionSet;

int h3 = H*H*H;
int h2 = H*H;
int s = 3;
int t = 4;

string fileLocate = "C:\\Users\\kimhyeji\\Desktop\\UCI\\project\\Pramook_black_velvet_3.03um_80kV_TIFF\\";
/*
open image files and store to 3D volume
*/
void makeVolume(vector<Mat> &v) {
	for (int i = 1; i <= IMG_NUM; i++) {
		Mat img;
		string image = fileLocate + format("%04d.tiff", i);
		//img = imread(image, CV_LOAD_IMAGE_ANYCOLOR);
		img = imread(image, CV_LOAD_IMAGE_UNCHANGED);
		normalize(img, img, 0.0, 1, CV_MINMAX, CV_32F);
		threshold(img, img, 0.7, 1, THRESH_BINARY_INV);
		img.convertTo(img, CV_8U, 255, 0);
		v[i - 1] = img;
		//imshow("img", img);
		//waitKey(0);
		img.release();
	}
}

/*
get rotation matrix
z ,x , y 순서
*/
Mat rotationMatrix(float d, float cosTheta) {
	cosTheta = -cosTheta;

	Mat XR = (Mat_<float>(3, 3) <<
		sin(d),0, -cos(d),
		0, 1, 0,
		cos(d), 0, sin(d)
		);
	float sinTheta = 1 - cosTheta*cosTheta;
	Mat ZR = (Mat_<float>(3, 3) <<
		1, 0, 0,
		0, cosTheta, -sinTheta,
		0, sinTheta, cosTheta
		);

	return ZR*XR;
}

/*
make set of direction vectors
*/
void makedirectionSet(directionSet &v, vector<Mat> &R) {
	float M_PI2 = M_PI * 2;
	float intervalTheta = M_PI2 / (DIRNUM - 1);
	float intervalZ = 2.0 / (DIRNUM - 1);

	int count = 0;
	for (float z = -1.0; z <= 1.0; z += intervalZ) {
		for (float theta = 0; theta <= M_PI2; theta += intervalTheta) {
			float x = sqrt(1.0 - pow(z, 2)) * cos(theta);
			float y = sqrt(1.0 - pow(z, 2)) * sin(theta);
			if (abs(z) == 1 && theta != 0) continue;
			R[count] = rotationMatrix(theta, z);
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
void saveDistance(float(&saveData)[H*H*H], const directionSet &d, vector<Mat> &R) {
	int h_half = H / 2;

	for (int p_z = 0; p_z < H; p_z++) {
		for (int p_x = 0; p_x < H; p_x++) {
			for (int p_y = 0; p_y < H; p_y++) {
				float dist = getDistance(Vec3f(1, 0, 0), Vec3f(p_z - h_half, p_x - h_half, p_y - h_half));
				float pow_dist = dist * dist;
				float q_func = -2 * exp(-s * pow_dist) + exp(-t * pow_dist);
				saveData[p_z*h2 + p_x*H + p_y] = q_func;
			}
		}
	}
}

/*
get Maximum convolution value
*/
Vec3f getMaxConvolution(const directionSet &d, const vector<Mat> &volume, const Vec3i &v, const float(&q)[H * H * H], bool density, vector<Mat> &R) {
	int x = v[1];
	int y = v[2];
	int z = v[0];
	float maxValue = -99999;
	int h_half = H / 2;

	Vec3f result;
	for (int i = 0; i < DIRNUMSQU; i++) {
		float j = 0;
		for (int p_z = 0; p_z < H; p_z++) {
			if (z + p_z >= IMG_NUM) continue;
			for (int p_x = 0; p_x < H; p_x++) {
				if (x + p_x >= ROW) continue;
				for (int p_y = 0; p_y < H; p_y++) {
					if (y + p_y >= COL) continue;
					Mat x_vec = (Mat_<float>(3, 1) <<
						p_z - h_half, p_x - h_half, p_y - h_half);
					Mat p = R[i] * x_vec;
					int v_x = int(x + p.at<float>(1, 0) + 0.5);
					int v_y = int(y + p.at<float>(2, 0) + 0.5);
					int v_z = int(z + p.at<float>(0, 0) + 0.5);

					if (v_z < 0) continue;
					if (volume[v_z].at<uchar>(v_x, v_y) > 0) {
						float q_func = q[p_z*h2 + p_x*H + p_y];
						j += q_func;
					}

				}
			}
		}
		// get maximum of J values
		if (maxValue < j) {
			maxValue = j;
			result = d[i];
		}
	}

	density = (maxValue > 0.001) ? 1 : 0;
	return result;
}

int main(int argc, char **argv) {

	vector<Mat> volumeData(IMG_NUM);
	vector<Mat> R(DIRNUMSQU, Mat(3, 3, CV_32FC1));
	float qfuncData[H*H*H];
	makeVolume(volumeData);

	directionSet dicVec(DIRNUMSQU);
	makedirectionSet(dicVec, R);
	saveDistance(qfuncData, dicVec, R);

	/*
	recover the orientation field
	*/
	for (int z = 0; z < IMG_NUM; z++) {
		cout << z << endl;
		Mat image_dir(ROW, COL, CV_32FC3);
		Mat image_den(ROW, COL, CV_32FC1);
		bool density = 0;

		#pragma omp parallel for schedule(dynamic,6)
		for (int x = 0; x < ROW; x++) {
			for (int y = 0; y < COL; y++) {
				if (volumeData[z].at<uchar>(x, y) > 0) {
					image_dir.at<Vec3f>(x, y)[0] = 0;
					image_dir.at<Vec3f>(x, y)[1] = 0;
					image_dir.at<Vec3f>(x, y)[2] = 0;
					continue;
				}
				Vec3f result = getMaxConvolution(dicVec, volumeData, Vec3i(z, x, y), qfuncData, density, R);

				image_dir.at<Vec3f>(x, y)[0] = result[1] * 255;//b
				image_dir.at<Vec3f>(x, y)[1] = result[2] * 255;//g
				image_dir.at<Vec3f>(x, y)[2] = result[0] * 255;//r
				if (density)
					image_den.at<float>(x, y) = volumeData[z].at<uchar>(x, y);
				else
					image_den.at<float>(x, y) = 0;

			}

			if (x % 100 == 0) {
				cout << x << endl;
				imwrite("zx" + format("%d.jpg", z), image_dir);
			}
		}
		imwrite("direction" + format("%d.tiff", z), image_dir);
		imwrite("density" + format("%d.tiff", z), image_den);
		image_dir.release();
		image_den.release();
	}

}


