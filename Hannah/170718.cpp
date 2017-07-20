#include "opencv2/opencv.hpp"

#include <iostream>
#include <cstdio>
#include <cmath>
#include <ctime>

using namespace std;
using namespace cv;
#define THRESHOLD 0.6
#define PI 3.141592
#define s 3
#define t 4
#define d 500
#define IMG_NUM 3 // 990
#define WIDTH  988 // 988
#define HEIGHT 1010 // 1010

class Direction {
public:
	float x = 0;
	float y = 0;
	float z = 0;
};

Direction unit_vec[1000];
double unit_vec_size = 0;
int vec_count = 0;
int cal_orientation();

int cal_binary_volume(Mat& mt_img);
float f(float x, float y, float z, vector<Mat> v_img_data);
int read_input(int file_num, vector<Mat>& v_img_data);




int main()
{
	clock_t begin = clock();
	//int width = 988, height = 1010;

	//Mat mt_img;
	//mt_img = imread("C:\\UCI\\Pramook_black_velvet_3.03um_80kV_TIFF\\Pramook_black_velvet_3.03um_80kV_TIFF\\0005.tiff", 0);


	vector<Mat> v_imgs;

	if (!read_input(IMG_NUM, v_imgs)) {
		cout << "image read error" << endl;
	}
	
	//imshow("으악",v_imgs[0]);
	//float a = mt_img.at<uchar>(560, 780);
	//float b = v_imgs[4].at<uchar>(560, 780);
	//cout << a << " : ";	cout << b << endl;

	// orientation vertor calculation
	cal_orientation();

	// function J calculation
	float binary_volume = 0;

	Mat d_volume(HEIGHT, WIDTH, CV_8UC3);
	int j_count = 0;
	float max_j = 0;
	float val_j = 0;
	Direction max_d;

	// cal_binary_volume(v_imgs[0]); // test


	
	for (int y_idx = 500; y_idx < 600; ++y_idx) { // height
		cout << "y_idx : " << y_idx << endl;
		for (int x_idx = 500; x_idx < 600; ++x_idx) { // width
			for (int z_idx = 500; z_idx < 600; ++z_idx) { // IMG_NUM

				if (f(x_idx, y_idx, z_idx, v_imgs)) {
					max_d.x = 0;
					max_d.y = 0;
					max_d.z = 0;
					continue;
				}

				max_j = 0;
				val_j = 0;
				max_d.x = 0;
				max_d.y = 0;
				max_d.z = 0;

				for (int p_x = -6; p_x <= 6; p_x++) {
					for (int p_y = -6; p_y <= 6; p_y++) {
						for (int p_z = -6; p_z <= 6; p_z++) {

							//cout << "thread Id : " << omp_get_thread_num() << endl;
							binary_volume = f(p_x + x_idx, p_y + y_idx, p_z + z_idx, v_imgs);
							//cout << "binary_volume(98) : " << binary_volume << endl;

							for (int i = 0; i < vec_count; i++) {
								float val_dot = (p_x*unit_vec[i].x + p_y*unit_vec[i].y + p_z*unit_vec[i].z);
								float tmp_x = p_x - val_dot*unit_vec[i].x;
								float tmp_y = p_y - val_dot*unit_vec[i].y;
								float tmp_z = p_z - val_dot*unit_vec[i].z;
								float r = abs(tmp_x*tmp_x + tmp_y*tmp_y + tmp_z*tmp_z); // || P - (P*d)*d ||
								float q = -2 * exp(-s*r) + exp(-t*r);

								//cout << "r(70) : " << r << endl;
								//cout << "q : " << q << endl;
								val_j += binary_volume*q;

								//cout << "val_j(78) : " << val_j << endl;
								if (val_j > max_j) {
									max_j = val_j;
									max_d.x = unit_vec[i].x;
									max_d.y = unit_vec[i].y;
									max_d.z = unit_vec[i].z;

									cout << "max_j (85) : " << max_j << endl;
									cout << "max_d : (" << max_d.x << ", " << max_d.y << ", " << max_d.z << ")" << endl;
								}
							}


						}
					}
				}

				d_volume.at<Vec3b>(y_idx, x_idx)[0] = max_d.x * 255; // blue
				d_volume.at<Vec3b>(y_idx, x_idx)[1] = max_d.y * 255; // green 
				d_volume.at<Vec3b>(y_idx, x_idx)[2] = max_d.z * 255; // red
			}
		}
	}
	imshow("debug4", d_volume);
	

	
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	cout << "Processing time: " << time_spent << "s" << endl;

	waitKey(0);
	return 0;
}


int cal_orientation() {
	float x = 0, y = 0, z = 0;
	for (int z_ = -10; z_ <= 10; z_ += 1) { // x10
		for (int theta = 0; theta <= 3600; theta += 360) { // x10

			float z = (float)z_ / 10;
			float degree = (float)theta / 10;

			float sin_theta = sin(degree*(PI / 180));
			float cos_theta = cos(degree*(PI / 180));
			//unit_vec_x[vec_count] = sqrt(1 - z*z)*cos_theta;
			//unit_vec_y[vec_count] = sqrt(1 - z*z)*sin_theta;
			//unit_vec_z[vec_count] = z;
			unit_vec[vec_count].x = sqrt(1 - z*z)*cos_theta;
			unit_vec[vec_count].y = sqrt(1 - z*z)*sin_theta;
			unit_vec[vec_count].z = z;
			vec_count++;
		}
	}
	return 1;
}

int cal_binary_volume(Mat& mt_img) {

	Mat binary_volume = mt_img;
	double bin = 0;
	cout << "debug1" << endl;
	for (int y_idx = 0; y_idx < mt_img.rows; ++y_idx) {
		for (int x_idx = 0; x_idx < mt_img.cols; ++x_idx) {
			bin = mt_img.at<uchar>(y_idx, x_idx);
			bin /= (float)255;
			if (bin >= THRESHOLD) binary_volume.at<uchar>(y_idx, x_idx) = 0;
			else binary_volume.at<uchar>(y_idx, x_idx) = 255;
		}
	}
	imshow("debug2",binary_volume);
	return 1;
}

float f(float x, float y, float z, vector<Mat> v_img_data) {

	double bin = 0;

	if (x >= v_img_data[0].cols || x < 0 || y >= v_img_data[0].rows || y <0 || z >= IMG_NUM || z<0) {
		return 0;
		cout << "187" << endl;
	}

	bin = v_img_data[z].at<uchar>(y, x);
	bin /= (float)255;

	//cout << "194" << endl;
	if (bin >= THRESHOLD) {
		return 0;
	}
	else {
		return 1;
	}
}


int read_input(int file_num, vector<Mat>& v_img_data) {

	Mat input;
	
	char cur_file_name[5];

	// store input data
	for (int cur_file_num = 1; cur_file_num <= file_num; cur_file_num++) {
		sprintf(cur_file_name, "%04d", cur_file_num);
		string cur_file_string(cur_file_name);

		// get next readed filename
		string path = "./";

		string filename = path + cur_file_string + ".tiff";
		//cout << "Opening image = " << filename << endl;


		//read image
		input = imread(filename, 0);
		if (input.empty()) return 0;

		//convert uchar to float -> 이거 하면 오류남
		//input.convertTo(input, CV_32FC1);


		v_img_data.push_back(input);

		//cout << "Image dimensions = " << input.size() << endl;
	}

	return 1;
}