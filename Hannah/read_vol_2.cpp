#include <cstdio>
#include <cstdint>
#include <stdio.h>
#include <iostream>
using namespace std;


float bytesToFloat(unsigned char b0, unsigned char b1, unsigned char b2, unsigned char b3) //little endian
{
	float output;
	*((unsigned char*)(&output) + 3) = b0;
	*((unsigned char*)(&output) + 2) = b1;
	*((unsigned char*)(&output) + 1) = b2;
	*((unsigned char*)(&output) + 0) = b3;
	return output;
}

int main(void)
{
	FILE    *fp_sour;
	FILE    *fp_dest;
	unsigned char buff[2048];
	size_t   n_size;
	char c;


	fp_sour = fopen("C:\\UCI\\Pramook_black_velvet_3.03um_80kV_down\\Pramook_black_velvet_3.03um_80kV_down.vol", "rb");
	n_size = fread(buff, 1, 2048, fp_sour);
	//    for(int i=0; i<n_size; i++){
	//        printf("%d: %d\n", i+1 ,buff[i]);
	//    }
	int i = 0;
	printf("%d: %c\n", i + 1, buff[i]);
	i++;
	printf("%d: %c\n", i + 1, buff[i]);
	i++;
	printf("%d: %c\n", i + 1, buff[i]);
	i++;
	printf("File format version : %d\n", buff[i]);
	i++;

	int final;

	final = 0;
	final |= ((int)(buff[i++]));
	final |= ((int)(buff[i++]) << 8);
	final |= ((int)(buff[i++]) << 16);
	final |= ((int)(buff[i++]) << 24);
	printf("%d: %d\n", i + 1, final);


	final = 0;
	final |= ((int)(buff[i++]));
	final |= ((int)(buff[i++]) << 8);
	final |= ((int)(buff[i++]) << 16);
	final |= ((int)(buff[i++]) << 24);
	printf("number of x: %d\n", final);


	final = 0;
	final |= ((int)(buff[i++]));
	final |= ((int)(buff[i++]) << 8);
	final |= ((int)(buff[i++]) << 16);
	final |= ((int)(buff[i++]) << 24);
	printf("number of y: %d\n", final);

	final = 0;
	final |= ((int)(buff[i++]));
	final |= ((int)(buff[i++]) << 8);
	final |= ((int)(buff[i++]) << 16);
	final |= ((int)(buff[i++]) << 24);
	printf("number of z: %d\n", final);

	final = 0;
	final |= ((int)(buff[i++]));
	final |= ((int)(buff[i++]) << 8);
	final |= ((int)(buff[i++]) << 16);
	final |= ((int)(buff[i++]) << 24);
	printf("number of channels: %d\n", final);

	////////////////////////////////////////////////////////

	float floatNum = 0;
	
	floatNum = bytesToFloat(buff[i++], buff[i++], buff[i++], buff[i++]);
	cout << "min x : " << floatNum << endl;
	floatNum = bytesToFloat(buff[i++], buff[i++], buff[i++], buff[i++]);
	cout << "min y : " << floatNum << endl;
	floatNum = bytesToFloat(buff[i++], buff[i++], buff[i++], buff[i++]);
	cout << "min z : " << floatNum << endl;
	floatNum = bytesToFloat(buff[i++], buff[i++], buff[i++], buff[i++]);
	cout << "max x : " << floatNum << endl;
	floatNum = bytesToFloat(buff[i++], buff[i++], buff[i++], buff[i++]);
	cout << "max y : " << floatNum << endl;
	floatNum = bytesToFloat(buff[i++], buff[i++], buff[i++], buff[i++]);
	cout << "max z : " << floatNum << endl;

	//////////////////////////////////////////////////////////

	fclose(fp_sour);
	return 0;
}