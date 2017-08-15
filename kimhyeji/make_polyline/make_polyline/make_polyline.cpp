/*
making Polyline using extracted information
*/

#include "read_VOL.h"
#include <iostream>
#include <time.h>
using namespace std;
using namespace cv;
class POLYLINE {
private:
	int IMGNUM, ROW, COL;
	float threshold_blob;
	bool threshold_check;

	vector<Mat> xVol;
	vector<Mat> yVol;
	vector<Mat> zVol;

	vector<vector<KeyPoint>> xBlob;
	vector<vector<KeyPoint>> yBlob;
	vector<vector<KeyPoint>> zBlob;

	vector<float> volume;

public:
	POLYLINE(char* locate, bool threshold_check = 1) {};
	
	//Trackbar for deciding the threshold
	static void onChange(int pos, void *user);
	float setThreshold(Mat image);
	void thresholding();

	//Blob detect
	vector<KeyPoint> blobDetect(Mat &image);
	
	//divide by axis
	void divideVolume();

	//match with points and make polyline
	void makePolyline();

	Point3f findRgbData(Point3i pos) {
		int lookupValue = ((pos.z*sy + pos.y)*sx + pos.x)*channels;
		return Point3f(volume[lookupValue], volume[lookupValue + 1], volume[lookupValue + 2]); //rgb
	}
};

POLYLINE::POLYLINE(char* locate, bool threshold_check = 1) {
	clock_t begin, end;
	begin = clock() / CLOCKS_PER_SEC;

	Point3i size = saveVolume(locate, volume);
	COL = size.x, ROW = size.y, IMGNUM = size.z;
	threshold_blob = 0.72;

	xVol.resize(IMGNUM), yVol.resize(IMGNUM), zVol.resize(IMGNUM);
	xBlob.resize(IMGNUM), yBlob.resize(IMGNUM), zBlob.resize(IMGNUM);
	end = clock() / CLOCKS_PER_SEC;
	printf("finished save : %d\n", (end - begin));
}

void POLYLINE::onChange(int pos, void *user) {
	Mat image = ((Mat*)user)[0];
	vector<Mat> splited(3);
	split(image, splited);
	Mat test;

	//Blob detector
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create();
	vector<KeyPoint> keypoints;

	float thresh = pos * 0.01;
	for (int i = 0; i < 3; i++) {
		threshold(splited[i], splited[i], thresh, 1, CV_THRESH_BINARY);
		splited[i].convertTo(splited[i], CV_8UC1, 255, 0);
		detector->detect(splited[i], keypoints);

		drawKeypoints(splited[i], keypoints, splited[i], Scalar(255,255,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		cvtColor(splited[i],splited[i], COLOR_BGR2GRAY);
	}
	merge(splited, test);
	imshow("image", test);
	imshow("image1", splited[0]);
	imshow("image2", splited[1]);
	imshow("image3", splited[2]);

}
float POLYLINE::setThreshold(Mat image) {
	int pos = threshold_blob * 100;
	Mat test;
	imshow("image", image);
	onChange(pos, (void *)&image);
	createTrackbar("threshold", "image", &pos, 100, POLYLINE::onChange, (void*)&image);
	while (1) {
		int Key = waitKeyEx();
		//arrow key <-
		if (Key == 2424832) {
			pos--;
			onChange(pos, (void *)&image);
			setTrackbarPos("threshold", "image", pos);
		}
		//arrow key ->
		else if(Key == 2555904){
			pos++;
			onChange(pos, (void *)&image);
			setTrackbarPos("threshold", "image", pos);
		}
		else {
			return pos*0.01;
		}
			if (pos < 0 || pos > 255) pos = 0;
	}
	return pos*0.01;
}

vector<KeyPoint> POLYLINE::blobDetect(Mat &image) {
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create();
	vector<KeyPoint> keypoints;
	detector->detect(image, keypoints);

	return keypoints;
}

void POLYLINE::thresholding() {
	for (int k = 0; k < IMGNUM; k++) {
		if (k == 0 && threshold_check) {
			vector<Mat> xyzMerge = {xVol[k] , yVol[k], zVol[k]};
			Mat test;
			merge(xyzMerge, test);
			threshold_blob = setThreshold(test);
			printf("threshold : %f\n", threshold_blob);
			destroyAllWindows();
		}
		threshold(xVol[k], xVol[k], threshold_blob, 1, CV_THRESH_BINARY);
		threshold(yVol[k], yVol[k], threshold_blob, 1, CV_THRESH_BINARY);
		threshold(zVol[k], zVol[k], threshold_blob, 1, CV_THRESH_BINARY);
		xVol[k].convertTo(xVol[k], CV_8U,255,0);
		yVol[k].convertTo(yVol[k], CV_8U,255,0);
		zVol[k].convertTo(zVol[k], CV_8U,255,0);
		xBlob[k] = blobDetect(xVol[k]);
		yBlob[k] = blobDetect(yVol[k]);
		zBlob[k] = blobDetect(zVol[k]);
		
		if(k % 100 == 0)printf(">> %d\n", k);
	}
}
void POLYLINE::divideVolume() {
	clock_t begin, end;
	begin = clock() / CLOCKS_PER_SEC;
	for (int k = 0;k < IMGNUM; k++) {

		Mat xImg(ROW, COL, CV_32FC1, Scalar(0.0)), yImg(ROW, COL, CV_32FC1, Scalar(0.0)), zImg(ROW, COL, CV_32FC1, Scalar(0.0));
		for (int i = 0; i < ROW;i++) {
			for (int j = 0; j < COL; j++){
				Point3f data = findRgbData(Point3i( j,i,k ));

				//get max direction
				if (data.x < data.y) {
					if (data.y < data.z) {
						*zImg.ptr<float>(i, j) = data.z;
					}
					else *yImg.ptr<float>(i, j) = data.y;
				}
				else if(data.x > data.z) *xImg.ptr<float>(i, j) =data.x;
				else *zImg.ptr<float>(i, j) = data.z;
			}
		}
		xVol[k] = xImg;
		yVol[k] = yImg;
		zVol[k] = zImg;
	}
	end = clock() / CLOCKS_PER_SEC;
	printf("divide finish : %d\n", (end - begin));
}
void POLYLINE::makePolyline() {
	divideVolume();
	thresholding();
	imshow("imageX.jpg", xVol[0]);
	imshow("imageY.jpg", yVol[0]);
	imshow("imageZ.jpg", zVol[0]);
	waitKey(0);

}

int main() {
	
	clock_t begin, end;
	begin = clock();
	char * fileLocation = "C:\\Users\\kimhyeji\\Desktop\\UCI\\project\\3-2_dir_down.vol";
	
	POLYLINE p(fileLocation);
	p.makePolyline();
	end = clock();

	printf("time : %d", (end - begin)/CLOCKS_PER_SEC);
}