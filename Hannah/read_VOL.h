#pragma once
#ifndef __READ_VOL_H
#define __READ_VOL_H

#include <cstdio>
#include <cstdint>
#include <stdio.h>
#include <iostream>
#include<vector>
#include "opencv2/opencv.hpp"

extern int sx, sy, sz;
extern int channels;

using namespace std;
using namespace cv;



float bytesToFloat(unsigned char b0, unsigned char b1, unsigned char b2, unsigned char b3);
void readHeader(unsigned char* buff);
void readData(vector<float> &data, FILE *fp_sour, int channels);
void printData(vector<float> &data);
float findData(vector<float> &data, Point3i pos);
Point3i findRgbData(vector<float> &data, Point3i pos);

#endif