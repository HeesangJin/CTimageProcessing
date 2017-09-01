
/*
#include <GL/freeglut.h>
#include <GL/GL.h>

#include <stdlib.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>

#include "read_VOL.h"

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bg::model::point<float, 2, bg::cs::cartesian> point;
typedef std::pair<point, unsigned> value;
typedef vector<vector<pair<Point2f, Point2f>>> POLYLINES;
POLYLINES polyline(3);

void myDisplay(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();//load the identity matrix
	gluLookAt(10.0f, 0.0f, 0.0f,
		0.0f, 0.0f, -10.0f,
		0.0f, 1.0f, 0.0f);
	glLineWidth(1.0);
	glBegin(GL_LINES);
	for (int i = 0; i < polyline.size(); i++) {
		for (int j = 0; j < polyline[i].size(); j++) {
			glColor3f(1, 0.5 + 0.1*i, 0.0);//colour added
			Point2f p = polyline[i][j].first;
			Point2f n_p = polyline[i][j].second;
			glVertex3f(p.x, p.y, i);
			glVertex3f(n_p.x, n_p.y, i+100);
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
	glOrtho(0.0, 1000.0, 0.0, 1000.0, -0.01, -80.0);//set the max mins for x,y and z
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

}

int main(int argc, char *argv[])
{
	vector<Mat> img(4);
	vector<vector<KeyPoint>> keypoints(4);
	string fileD = "C:\\Users\\kimhyeji\\Desktop\\CTimageProcessing\\kimhyeji\\make_polyline\\make_polyline";
	img[0] = imread(fileD + "\\y0.jpg", 0);
	img[1] = imread(fileD + "\\y1.jpg", 0);
	img[2] = imread(fileD + "\\y2.jpg", 0);
	img[3] = imread(fileD + "\\y3.jpg", 0);

	//Blob detector
	SimpleBlobDetector::Params params;
	params.minArea = 0.1;
	params.filterByInertia = false;
	params.filterByConvexity = false;
	params.filterByColor = false;
	params.filterByCircularity = false;
	params.filterByArea = true;
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	for (int i = 0; i < 4; i++) {
		threshold(img[i], img[i], 100, 255, CV_THRESH_BINARY);
		detector->detect(img[i], keypoints[i]);
	}


	// create tree
	bgi::rtree< value, bgi::quadratic<16> > RT;
	for (unsigned i = 0; i < keypoints[0].size(); ++i) {
		point p = point(keypoints[0][i].pt.x, keypoints[0][i].pt.y);
		RT.insert(std::make_pair(p, i));
	}

	for (int j = 0; j < 3; j++) {
		bgi::rtree< value, bgi::quadratic<16> > nextRT;

		// search for nearest neighbours
		std::vector<value> matchPoints;
		vector<pair<float, float>> pointList;

		for (unsigned i = 0; i < keypoints[j + 1].size(); ++i) {
			point p = point(keypoints[j + 1][i].pt.x, keypoints[j + 1][i].pt.y);
			nextRT.insert(std::make_pair(p, i));
			RT.query(bgi::nearest(p, 1), std::back_inserter(matchPoints));

			if (bg::distance(p, matchPoints.back().first) > 4) matchPoints.pop_back();
			else {
				pointList.push_back(make_pair(keypoints[j + 1][i].pt.x, keypoints[j + 1][i].pt.y));
				RT.remove(matchPoints.back());
			}
		}
		printf("%d > %d\n", matchPoints.size(), pointList.size());

		// print returned values
		value nextPoint;
		for (size_t i = 0; i < matchPoints.size(); i++) {
			nextPoint = matchPoints[i];
			float n_x = nextPoint.first.get<0>();
			float n_y = nextPoint.first.get<1>();
			float x = pointList[i].first;
			float y = pointList[i].second;
			polyline[j].push_back(make_pair( Point2f(x,y),Point2f(n_x,n_y) ) );
		}
		RT = nextRT;

	}
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(600, 600);
	glutInitWindowPosition(100, 150);
	glutCreateWindow("3D line");
	
	glutDisplayFunc(myDisplay);
	glEnable(GL_DEPTH_TEST);
	initializeGL();
	glutMainLoop();


	return 0;
}
*/