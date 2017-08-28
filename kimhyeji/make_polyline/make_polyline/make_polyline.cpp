/*
making Polyline using extracted information
*/
#include <GL/freeglut.h>
#include <GL/GL.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>

#include "read_VOL.h"
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <time.h>
#include <boost/geometry.hpp>

#include <omp.h>

using namespace std;
using namespace cv;
int countP = 0;

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bg::model::point<float, 2, bg::cs::cartesian> point;
typedef std::pair<point, unsigned> value;
typedef vector<vector<KeyPoint>> POLY;
typedef vector<vector<pair<Point2f, Point2f>>> CONNECTPOLY;
CONNECTPOLY polyline;

class POLYLINE {
private:
	int IMGNUM, ROW, COL;
	float threshold_blob;
	bool threshold_check;

	vector<Mat> xVol;
	vector<Mat> yVol;
	vector<Mat> zVol;

	POLY xBlob;
	POLY yBlob;
	POLY zBlob;

	vector<float> volume;

public:
	POLYLINE(char* locate, bool threshold_check_);
	
	float absData(float f);

	//Trackbar for deciding the threshold
	static void onChange(int pos, void *user);
	float setThreshold(vector<Mat> image);
	void thresholding();

	//Blob detect
	vector<KeyPoint> blobDetect(Mat &image);
	
	//divide by axis
	void divideVolume();

	//match with points and make polyline
	void connectDot(const POLY &v);
	void makePolyline();

	void printDot(char* fileName);

	Point3f findRgbData(Point3i pos) {
		int lookupValue = ((pos.z*sy + pos.y)*sx + pos.x)*channels;
		return Point3f(volume[lookupValue], volume[lookupValue + 1], volume[lookupValue + 2]); //rgb
	}
};

POLYLINE::POLYLINE(char* locate, bool threshold_check_ = 1) {
	clock_t begin, end;
	begin = clock() / CLOCKS_PER_SEC;

	threshold_check = threshold_check_;
	Point3i size = saveVolume(locate, volume);
	COL = size.x, ROW = size.y, IMGNUM = size.z;
	threshold_blob = 0.72;
	xVol.resize(COL), yVol.resize(ROW), zVol.resize(IMGNUM);
	xBlob.resize(COL), yBlob.resize(ROW), zBlob.resize(IMGNUM);
	end = clock() / CLOCKS_PER_SEC;
	printf("finished save : %d\n", (end - begin));
}

void POLYLINE::onChange(int pos, void *user) {
	vector<Mat> s = ((vector<Mat>*)user)[0];
	vector<Mat> splited(3);
	for(int i = 0; i < 3; i++) 	s[i].copyTo(splited[i]);
	
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

	imshow("image", splited[0]);
	imshow("x", splited[1]);
	imshow("y", splited[2]);
}

float POLYLINE::setThreshold(vector<Mat> image) {
	int pos = threshold_blob * 100;
	imshow("image", image[0]);
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
	vector<KeyPoint> keypoints;
	SimpleBlobDetector::Params params;
	params.minArea = 0.1;
	params.filterByInertia = false;
	params.filterByConvexity = false;
	params.filterByColor = false;
	params.filterByCircularity = false;
	params.filterByArea = true;
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	
	detector->detect(image, keypoints);
	return keypoints;
}

void POLYLINE::thresholding() {
	#pragma omp parallel for schedule(dynamic,6)
	for (int k = 0; k < IMGNUM; k++) {
		if (threshold_check && k < ROW && k < COL) {
			vector<Mat> xyzMerge = {xVol[k] , yVol[k], zVol[k]};
			threshold_blob = setThreshold(xyzMerge);
			printf("threshold : %f\n", threshold_blob);
			destroyAllWindows();
		}
		threshold(zVol[k], zVol[k], threshold_blob, 1, CV_THRESH_BINARY);
		zVol[k].convertTo(zVol[k], CV_8U,255,0);
		zBlob[k] = blobDetect(zVol[k]);	

		if (k % 100 == 0)  printf(" >");
	}
	printf("z\n");
	
	#pragma omp parallel for schedule(dynamic,6)
	for (int k = 0; k < ROW; k++) {
		threshold(yVol[k], yVol[k], threshold_blob, 1, CV_THRESH_BINARY);
		yVol[k].convertTo(yVol[k], CV_8U, 255, 0);
		yBlob[k] = blobDetect(yVol[k]);
	}
	printf("y\n");

	#pragma omp parallel for schedule(dynamic,6)
	for (int k = 0; k < COL; k++) {
		threshold(xVol[k], xVol[k], threshold_blob, 1, CV_THRESH_BINARY);
		xVol[k].convertTo(xVol[k], CV_8U, 255, 0);
		xBlob[k] = blobDetect(xVol[k]);
	}
	printf("x\n");
	
}
void POLYLINE::connectDot(const POLY &keypoints) {
	polyline.resize(keypoints.size());

	// create first RT tree
	bgi::rtree< value, bgi::quadratic<16> > RT;
	for (unsigned i = 0; i < keypoints[0].size(); ++i) {
		point p = point(keypoints[0][i].pt.x, keypoints[0][i].pt.y);
		RT.insert(std::make_pair(p, i));
	}
	printf("make RT\n");

	for (int j = 0; j < keypoints.size()-1; j++) {
		bgi::rtree< value, bgi::quadratic<16> > nextRT;

		// search for nearest neighbours
		std::vector<value> matchPoints;
		vector<pair<float, float>> pointList;

		for (unsigned i = 0; i < keypoints[j + 1].size(); ++i) {
			point p = point(keypoints[j + 1][i].pt.x, keypoints[j + 1][i].pt.y);
			nextRT.insert(std::make_pair(p, i));
			RT.query(bgi::nearest(p, 1), std::back_inserter(matchPoints));

			// within 4 distance
			if (bg::distance(p, matchPoints.back().first) > 4) matchPoints.pop_back();
			else {
				pointList.push_back(make_pair(keypoints[j + 1][i].pt.x, keypoints[j + 1][i].pt.y));
				RT.remove(matchPoints.back());
			}
		}

		// print returned values
		value nextPoint;
		for (size_t i = 0; i < matchPoints.size(); i++) {
			nextPoint = matchPoints[i];
			float n_x = nextPoint.first.get<0>();
			float n_y = nextPoint.first.get<1>();
			float x = pointList[i].first;
			float y = pointList[i].second;
			polyline[j].push_back(make_pair(Point2f(x, y), Point2f(n_x, n_y)));
		}
		RT = nextRT;
		if (j % 100 == 0)  printf("%d >", j);
	}
}
float POLYLINE::absData(float p) {
	if (p < 0) p = -p;
	return p;
}

void POLYLINE::divideVolume() {
	clock_t begin, end;
	begin = clock() / CLOCKS_PER_SEC;
	//initialize
	for (int k = 0;k < COL; k++) xVol[k] = Mat(ROW, IMGNUM, CV_32FC1, Scalar(0.0));
	for(int k = 0; k < ROW; k++) yVol[k] = Mat(IMGNUM, COL, CV_32FC1, Scalar(0.0));
	for(int k = 0; k < IMGNUM;k++)	zVol[k] = Mat(ROW, COL, CV_32FC1, Scalar(0.0));
	
	for (int k = 0;k < IMGNUM; k++) {
		#pragma omp parallel for schedule(dynamic,4)
		for (int i = 0; i < ROW;i++) {
			for (int j = 0; j < COL; j++){
				Point3f data = findRgbData( Point3i( j,i,k ) );
				data = Point3f(absData(data.x), absData(data.y), absData(data.z));
				
				//get max direction
				if (data.x < data.y) {
					if (data.y < data.z) {
						*zVol[k].ptr<float>(i, j) = data.z;
					}

					else *yVol[i].ptr<float>(IMGNUM-k-1, j) = data.y;
				}
				else if(data.x > data.z) *xVol[j].ptr<float>(i, IMGNUM-k-1) = data.x;
				else *zVol[k].ptr<float>(i, j) = data.z;

			}
		}
	}
	end = clock() / CLOCKS_PER_SEC;
	printf("divide finish : %d\n", (end - begin));
}

void POLYLINE::makePolyline() {
	divideVolume();
	thresholding();
	connectDot(zBlob);
	printDot("zData.txt");
	connectDot(yBlob);
	printDot("yData.txt");
	connectDot(xBlob);
	printDot("xData.txt");



	//connectDot(yBlob);

}

void POLYLINE::printDot(char * fileName) {

	ofstream output;
	output.open(fileName);
	printf(">>%d\n", polyline.size());
	for (int i = 0; i < polyline.size(); i++) {
		output << i << endl;
		printf("%d>>>%d  ", i, polyline[i].size());

		for (int j = 0; j < polyline[i].size(); j++) {
			output << polyline[i][j].first.x << ' ' << polyline[i][j].first.y << ' ';
			output << polyline[i][j].second.x << ' ' << polyline[i][j].second.y << endl;
		}
	}
}

void myDisplay(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();//load the identity matrix
	gluLookAt(100.0f, 0.0f, 0.0f,
		0.0f, 0.0f, -100.0f,
		0.0f, 1.0f, 0.0f);
	glLineWidth(1.0);
	glBegin(GL_LINES);
	for (int i = 0; i < polyline.size(); i++) {
		for (int j = 0; j < polyline[i].size(); j++) {
			glColor3f(1, 0.5 + 0.1*i, 1 - 0.1*i);//colour added
			Point2f p = polyline[i][j].first;
			Point2f n_p = polyline[i][j].second;
			glVertex3f(p.x, p.y, i);
			glVertex3f(n_p.x, n_p.y, i + 1);
		}
	}
	glEnd();
	glutPostRedisplay();
	glutSwapBuffers();
}
void initializeGL(void)
{
	glFlush();//initialize the screen before adding any elements
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glClearColor(0, 0, 0, 1.0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 600.0, 0.0, 600.0, -0.01, -600.0);//set the max mins for x,y and z
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

}

int main(int argc, char *argv[]) {
	
	clock_t begin, end;
	begin = clock();
	char * fileLocation = "C:\\Users\\kimhyeji\\Desktop\\UCI\\project\\3-2_dir_down.vol";
	
	POLYLINE p(fileLocation,0);
	p.makePolyline();
	
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(600, 600);
	glutInitWindowPosition(100, 150);
	glutCreateWindow("3D line");

	glutDisplayFunc(myDisplay);
	glEnable(GL_DEPTH_TEST);
	initializeGL();
	glutMainLoop();

	end = clock();
	printf("time : %d", (end - begin)/CLOCKS_PER_SEC);
}

