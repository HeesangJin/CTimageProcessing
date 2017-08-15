#pragma once
#ifndef __READ_VOL_H
#define __READ_VOL_H

#include "opencv2/opencv.hpp"
#include <cstdio>
#include <cstdint>
#include <stdio.h>
#include <iostream>
#include <vector>

extern int sx, sy, sz;
extern int channels;

using namespace std;
using namespace cv;

float bytesToFloat(unsigned char b0, unsigned char b1, unsigned char b2, unsigned char b3);
void readHeader(unsigned char* buff);
void readData(vector<float> &data, FILE *fp_sour);
Point3f saveVolume(char* fileLocation, vector<float> &volume);
void printData(vector<float> &data);
float findData(vector<float> &data, Point3i pos);
Point3f findRgbData(vector<float> &data, Point3i pos);

#endif