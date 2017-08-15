#include "read_VOL.h"

int final;
int channels = 0;
int idx;
int sx = 0, sy = 0, sz = 0;


float bytesToFloat(unsigned char b0, unsigned char b1, unsigned char b2, unsigned char b3) //little endian
{
	float output;
	*((unsigned char*)(&output) + 3) = b0;
	*((unsigned char*)(&output) + 2) = b1;
	*((unsigned char*)(&output) + 1) = b2;
	*((unsigned char*)(&output) + 0) = b3;
	return output;
}

void readHeader(unsigned char* buff) {

	//print VOL
	for (idx = 0; idx<3; idx++)	printf("%d: %c\n", idx + 1, buff[idx]);

	printf("File format version : %d\n", buff[idx++]);


	final = 0;
	final |= ((int)(buff[idx++]));
	final |= ((int)(buff[idx++]) << 8);
	final |= ((int)(buff[idx++]) << 16);
	final |= ((int)(buff[idx++]) << 24);
	printf("Encoding identifier: %d\n", final);


	final = 0;
	final |= ((int)(buff[idx++]));
	final |= ((int)(buff[idx++]) << 8);
	final |= ((int)(buff[idx++]) << 16);
	final |= ((int)(buff[idx++]) << 24);
	printf("number of x: %d\n", final);
	sx = final;

	final = 0;
	final |= ((int)(buff[idx++]));
	final |= ((int)(buff[idx++]) << 8);
	final |= ((int)(buff[idx++]) << 16);
	final |= ((int)(buff[idx++]) << 24);
	printf("number of y: %d\n", final);
	sy = final;

	final = 0;
	final |= ((int)(buff[idx++]));
	final |= ((int)(buff[idx++]) << 8);
	final |= ((int)(buff[idx++]) << 16);
	final |= ((int)(buff[idx++]) << 24);
	printf("number of z: %d\n", final);
	sz = final;

	final = 0;
	final |= ((int)(buff[idx++]));
	final |= ((int)(buff[idx++]) << 8);
	final |= ((int)(buff[idx++]) << 16);
	final |= ((int)(buff[idx++]) << 24);
	printf("number of channels: %d\n", final);
	channels = final;


	float floatNum = 0;

	floatNum = bytesToFloat(buff[idx++], buff[idx++], buff[idx++], buff[idx++]);
	cout << "min x : " << floatNum << endl;
	floatNum = bytesToFloat(buff[idx++], buff[idx++], buff[idx++], buff[idx++]);
	cout << "min y : " << floatNum << endl;
	floatNum = bytesToFloat(buff[idx++], buff[idx++], buff[idx++], buff[idx++]);
	cout << "min z : " << floatNum << endl;
	floatNum = bytesToFloat(buff[idx++], buff[idx++], buff[idx++], buff[idx++]);
	cout << "max x : " << floatNum << endl;
	floatNum = bytesToFloat(buff[idx++], buff[idx++], buff[idx++], buff[idx++]);
	cout << "max y : " << floatNum << endl;
	floatNum = bytesToFloat(buff[idx++], buff[idx++], buff[idx++], buff[idx++]);
	cout << "max z : " << floatNum << endl;

}

Point3f saveVolume(char* fileLocation, vector<float> &volume) {
	FILE    *fp_sour;
	size_t   n_size;
	unsigned char buff[48]; //48byte

	fp_sour = fopen(fileLocation, "rb");
	n_size = fread(buff, 1, 48, fp_sour);
	readHeader(buff);

	volume.resize(((((long long)sx)*sy)*sz)*channels );

	readData(volume, fp_sour);

	fclose(fp_sour);

	return Point3f(sx, sy, sz);

}

void readData(vector<float> &data, FILE *fp_sour) {

	int dataIdx = 0;

	size_t n_count = 0;
	n_count = fread(&data[dataIdx], sizeof(float), sx*sy*sz*channels, fp_sour); // sx*sy*sz * sizeof(float)

	cout << "n_count : " << n_count << endl;
	cout << "sx*sy*sz*channels : " << sx*sy*sz*channels << endl;
	if (n_count != sx*sy*sz*channels)
	{
		fputs("Reading error", stderr);
		exit(3);
	}

	cout << "data size: " << data.size() << endl << endl;
}

void printData(vector<float> &data) {
	for (int i = 0; i < data.size(); i++) {
		if (data[i])  printf("data[%d] = %f\n", i, data[i]);
		//else printf("data[%d] = %f\n", i, data[i]);
	}
}

float findData(vector<float> &data, Point3i pos) {

	/***********************************************************
	data[((zpos*yres + ypos)*xres + xpos)*channels + chan]
	where (xpos, ypos, zpos, chan) denotes the lookup location.
	************************************************************/

	int lookupValue = ((pos.z*sy + pos.y)*sx + pos.x)*channels;
	//cout << "(x,y,z) : " << "(" << pos.x << ", " << pos.y << ", " << pos.z << ")" << endl;
	//cout << "lookup value : " << lookupValue << endl;
	//cout << "Find data[" << lookupValue << "] : " << data[lookupValue] << endl;
	float lookupData = data[lookupValue];
	return lookupData;
}

Point3f findRgbData(vector<float> &data, Point3i pos) {

	/***********************************************************
	data[((zpos*yres + ypos)*xres + xpos)*channels + chan]
	where (xpos, ypos, zpos, chan) denotes the lookup location.
	************************************************************/

	int lookupValue = ((pos.z*sy + pos.y)*sx + pos.x)*channels;
	//cout << "(x,y,z) : " << "(" << pos.x << ", " << pos.y << ", " << pos.z << ")" << endl;
	//cout << "lookup value : " << lookupValue << endl;
	//cout << "Find data[" << lookupValue << "] : " << data[lookupValue] << endl;
	return Point3f(data[lookupValue],data[lookupValue + 1], data[lookupValue + 2]); //rgb
}