#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;

// kiem tra ảnh có phải là gray_scale
bool isGrayScale(const Mat& Img);

// công thức chuyển đổi ảnh màu sang xám
int formular1(int r, int g, int b);

// chuyển doi ảnh màu sang ảnh xám
void rgbToGray(Mat srcImg, Mat& dstImg);

// so sánh 2 histogram của ảnh
double compareHist(const Mat& srcImg1, Mat srcImg2);

// phân chia dử liệu điểm ảnh của 3 kênh màu
void split(const Mat &srcImg, int brg_planes[3][256]);

// tính 2 histogram của 2 ảnh lượng hóa
void calcHistogram(const Mat &srcImg, int rBin, int gBin, int bBin, int hist[3][256]);

// tình histogram của ảnh xám
void calcHistogram(const Mat &srcImg, int grayBin, int grayhist[256]);

// so sánh histogram của 2 ảnh lượng hóa
double compareHist(const Mat& srcImg1, const Mat& srcImg2, int rBin, int gBin, int bBin);

// so sánh 2 ảnh lượng hóa xám
double compareHist_g(const Mat& srcImg1, const Mat &srcImg2, int gBin);

// trả về ảnh đạo hàm theo hướng x
Mat xGradient(const Mat &src, double k, double coefficient);

// trả về ảnh đạo hàm theo hướng y
Mat yGradient(const Mat &src, double k, double coefficient);

// tính gradient tại 1 điểm ảnh
int xGrad(Mat image, int x, int y);
int yGrad(Mat image, int x, int y);

// tính magnitude cùa ảnh
Mat magnitude(const Mat &src, double k, double coeff);


////////////////////////////////////////

// kerneals
double Wx[3][3] = { { 1, 0, -1 },
{ 1, 0, -1 },
{ 1, 0, -1 } };

double Wy[3][3] = { { -1, -1, -1 },
{ 0, 0, 0 },
{ 1, 1, 1 } };

const float cr = 0.299, cg = 0.587, cb = 0.114;
// get type of Mat

//string type2str(int type) {
//	string r;
//
//	uchar depth = type & CV_MAT_DEPTH_MASK;
//	uchar chans = 1 + (type >> CV_CN_SHIFT);
//
//	switch (depth) {
//	case CV_8U:  r = "8U"; break;
//	case CV_8S:  r = "8S"; break;
//	case CV_16U: r = "16U"; break;
//	case CV_16S: r = "16S"; break;
//	case CV_32S: r = "32S"; break;
//	case CV_32F: r = "32F"; break;
//	case CV_64F: r = "64F"; break;
//	default:     r = "User"; break;
//	}
//
//	r += "C";
//	r += (chans + '0');
//
//	return r;
//}

bool isGrayScale(const Mat& Img) {
	int width = Img.cols;
	int height = Img.rows;

	if (Img.type() == CV_8UC1)
		return true;
	else
		return false;

	return true;
}

int formular1(int r, int g, int b) {
	float fr = r / 255.0;
	float fg = g / 255.0;
	float fb = b / 255.0;
	return (int)((cr  * fr + cg * fg + cb * fb)*255.0);
}

void rgbToGray(Mat srcImg, Mat& dstImg) {
	int with = srcImg.cols;
	int height = srcImg.rows;

	cout << with << " " << height << endl;
	//create new image
	dstImg.create(height, with, srcImg.type());


	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < with; ++j) {
			dstImg.at<Vec3b>(i, j).val[0] = dstImg.at<Vec3b>(i, j).val[1] = dstImg.at<Vec3b>(i, j).val[2] = formular1(srcImg.at<Vec3b>(j, i).val[2], srcImg.at<Vec3b>(j, i).val[1], srcImg.at<Vec3b>(j, i).val[0]);
		}
	}

	cout << "dstImg : " << dstImg.cols << " " << dstImg.rows << endl;
}

//
//void draw(const Mat& data, int r, int g, int b) {
//	Mat plot_result;
//	Ptr<plot::Plot2d> plot = cv::plot::Plot2d::create(data);
//	plot->setPlotBackgroundColor(Scalar(50, 50, 50));
//	plot->setPlotLineColor(Scalar(r, g, b));
//	plot->render(plot_result);
//	imshow("Graph", plot_result);
//}


// compare two histogram of color image
double compareHist(const Mat& srcImg1, Mat srcImg2) {

	int row1 = srcImg1.rows;
	int row2 = srcImg2.rows;
	int col1 = srcImg1.cols;
	int col2 = srcImg2.cols;
	int hist1[3][256];
	int hist2[3][256];
	calcHistogram(srcImg1, 1, 1, 1, hist1);
	calcHistogram(srcImg2, 1, 1, 1, hist2);

	double res = 0.0;

	vector<double> v1 = { 0.0, 0.0, 0.0 };
	vector<double> v2 = { 0.0, 0.0, 0.0 };

	for (int i = 0; i < 256; ++i) {
		res += sqrt(pow(hist1[0][i] - hist1[0][i], 2) + pow(hist1[1][i] - hist1[1][i], 2) + pow(hist1[2][i] - hist1[2][i], 2));
	}
	return res;
}

// splits image into R, G, B planes
void split(const Mat &srcImg, int brg_planes[3][256]) {
	memset(brg_planes, 0, 3 * 256 * sizeof(int));
	int rows = srcImg.rows;
	int cols = srcImg.cols;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			brg_planes[0][(int)(srcImg.at<Vec3b>(i, j).val[0])] += 1;
			brg_planes[1][(int)(srcImg.at<Vec3b>(i, j).val[1])] += 1;
			brg_planes[2][(int)(srcImg.at<Vec3b>(i, j).val[2])] += 1;
		}
	}
}

void calcHistogram(int plane[], int bin, int hist[256]) {

	int num = 256 / bin;
	for (int i = 0; i < 256; ++i) {
		hist[(i / num >= bin ? bin - 1 : i / num)] += plane[i];
	}
}

void calcHistogram(const Mat &srcImg, int rBin, int gBin, int bBin, int hist[3][256]) {
	// rhistogram

	int rgb_planes[3][256];


	split(srcImg, rgb_planes);

	calcHistogram(rgb_planes[2], rBin, hist[2]);
	calcHistogram(rgb_planes[1], gBin, hist[1]);
	calcHistogram(rgb_planes[0], bBin, hist[0]);
}

// tinh histogram luong hoa anh xam
void calcHistogram(const Mat &srcImg, int grayBin, int grayhist[256]) {

	int gnum = 256 / grayBin;
	int rows = srcImg.rows;
	int cols = srcImg.cols;

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			int pos = (int)(srcImg.at<uchar>(i, j)) / gnum;
			grayhist[(int)(pos >= grayBin ? grayBin - 1 : pos)] += 1;
		}
	}
	// draw histogram;
}

// so sanh hai anh luong hoa mau
double compareHist(const Mat& srcImg1, const Mat& srcImg2, int rBin, int gBin, int bBin) {
	int rhist[256];
	int ghist[256];
	int bhist[256];

	memset(rhist, 0, 256 * sizeof(int));
	memset(ghist, 0, 256 * sizeof(int));
	memset(bhist, 0, 256 * sizeof(int));

	int rgb_planes[3][256];
	split(srcImg1, rgb_planes);

	calcHistogram(rgb_planes[2], rBin, rhist);
	calcHistogram(rgb_planes[1], gBin, ghist);
	calcHistogram(rgb_planes[0], bBin, bhist);

	int rhist1[256];
	int ghist1[256];
	int bhist1[256];
	memset(rhist1, 0, 256 * sizeof(int));
	memset(ghist1, 0, 256 * sizeof(int));
	memset(bhist1, 0, 256 * sizeof(int));
	int rgb_planes1[3][256];
	split(srcImg1, rgb_planes1);

	calcHistogram(rgb_planes1[2], rBin, rhist1);
	calcHistogram(rgb_planes1[1], gBin, ghist1);
	calcHistogram(rgb_planes1[0], bBin, bhist1);

	double res = 0, res1 = 0, res2 = 0;

	for (int i = 0; i < rBin; ++i) {
		res += sqrt(pow(rhist[i] - rhist1[i], 2));
	}

	for (int i = 0; i < gBin; ++i) {
		res1 += sqrt(pow(ghist[i] - ghist1[i], 2));
	}

	for (int i = 0; i < bBin; ++i) {
		res2 += sqrt(pow(bhist[i] - bhist1[i], 2));
	}

	return (res1 + res2 + res) / 3.0;
}

// so sanh hai anh luong hoa xam
double compareHist_g(const Mat& srcImg1, const Mat &srcImg2, int gBin) {

	int grayhist1[256];
	memset(grayhist1, 0, sizeof(int) * 256);
	int grayhist2[256];
	memset(grayhist2, 0, sizeof(int) * 256);

	calcHistogram(srcImg1, gBin, grayhist1);

	calcHistogram(srcImg2, gBin, grayhist2);

	double res = 0;
	for (int i = 0; i < gBin; ++i) {
		res += sqrt(pow(grayhist1[i] - grayhist2[i], 2));
	}
	return res;
}



Mat xGradient(const Mat &src, double k, double coefficient) {
	int width = src.cols;
	int height = src.rows;

	Wx[1][0] = k; Wx[1][2] = k*(-1);
	Mat dstImg;
	dstImg.create(height, width, CV_8UC1);

	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			dstImg.at<uchar>(y, x) = 0.0;
	for (int y = 1; y < height - 1; ++y) {
		for (int x = 0; x < width - 1; ++x) {
			int gx = xGrad(src, x, y);
			gx = gx > 255 ? 255 : gx;
			gx = gx < 0 ? 0 : gx;
			dstImg.at<uchar>(y, x) = gx*coefficient;
		}
	}
	return dstImg;
}

// tinh dao ham theo huong y
Mat yGradient(const Mat &src, double k, double coefficient) {
	int width = src.cols;
	int height = src.rows;


	Wy[0][1] = k*(-1); Wy[2][1] = k;
	Mat dstImg = src.clone();

	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			dstImg.at<uchar>(y, x) = 0.0;
	for (int y = 1; y < height - 1; ++y) {
		for (int x = 1; x < width - 1; ++x) {
			int gy = yGrad(src, x, y)*coefficient;
			gy = gy > 255 ? 255 : gy;
			gy = gy < 0 ? 0 : gy;
			dstImg.at<uchar>(y, x) = gy;
		}
	}
	return dstImg;
}


// tinh dao ham hai huong theo x, y
Mat magnitude(const Mat &src, double k, double coeff) {

	Wy[0][1] = k*(-1); Wy[2][1] = k;
	Wx[1][0] = k; Wx[1][2] = k*(-1);

	int height = src.rows, width = src.cols;
	Mat dstImg = src.clone();

	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			dstImg.at<uchar>(y, x) = 0.0;

	for (int y = 1; y < src.rows - 1; y++) {
		for (int x = 1; x < src.cols - 1; x++) {
			int gx = xGrad(src, x, y);
			int gy = yGrad(src, x, y);
			int	sum = sqrt(pow(gx, 2) + pow(gy, 2)) / 4;
			sum = sum > 255 ? 255 : sum;
			sum = sum < 0 ? 0 : sum;
			dstImg.at<uchar>(y, x) = sum;
		}
	}
	return dstImg;
}

int xGrad(Mat image, int x, int y)
{
	int sum = 0;
	for (int i = -1; i <= 1; ++i)
		for (int j = -1; j <= 1; ++j) {
			sum += image.at<uchar>(y + i, x + j)*Wx[1 - i][1 - j];
		}
	return sum;
}

// Computes the y component of the gradient vector
// at a given point in a image
// returns gradient in the y direction

int yGrad(Mat image, int x, int y)
{
	int sum = 0;
	for (int i = -1; i <= 1; ++i)
		for (int j = -1; j <= 1; ++j) {
			sum += image.at<uchar>(y + i, x + j)*Wy[1 - i][1 - j];
		}
	return sum;
}


Mat reduce_intensity(Mat image, int factor) {
	return (image / factor);
}

Mat image, image1, image2;

int main(int argc, char **argv) {

	int ch;
	for (int i = 0; i < argc; ++i) {
		if (argv[i][0] >= '0' && argv[i][0] <= '9') {
			stringstream ss(argv[i]);
			ss >> ch;
			break;
		}
	}
	
	cout << ch << endl;
	switch (ch)
	{
	case 1: {
		if (argc < 2) {
			cout << "loi tham so" << endl;
			return -1;
		}
		image = imread(argv[1], IMREAD_ANYCOLOR);
		if (!image.data) {
			cout << "khong mo duoc anh " << endl;
			return -1;
		}

		if (isGrayScale(image)) {
			cout << "input anh la gray scale" << endl;
		}
		else {
			rgbToGray(image, image1);
			namedWindow("Display window", WINDOW_AUTOSIZE);
			imshow("Display window", image1);
		}
		break;
	}
	case 2: {
		if (argc < 2) {
			cout << "loi tham so" << endl;
			return -1;
		}
		image = imread(argv[1], IMREAD_ANYCOLOR);
		if (!image.data) {
			cout << "khong mo duoc anh " << endl;
			return -1;
		}

		if (isGrayScale(image)) {
			int grayhist[256];
			memset(grayhist, 0, 256 * sizeof(int));
			calcHistogram(image, 256, grayhist);
			cout << "histogram cua anh xam " << endl;
			for (int i = 0; i < 256; ++i) {
				cout << "level : " << i << " : " << grayhist[i] << endl;
			}
		}
		else {
			int hist[3][256];
			memset(hist, 0, sizeof(int) * 3 * 256);
			calcHistogram(image, 256, 256, 256, hist);
			cout << "histogram cua anh mau : " << endl;
			cout << "		Red    green    blue" << endl;
			for (int i = 0; i < 256; ++i) {
				cout << "level : " << i << " : " << hist[1][i] << " " << hist[2][i] << " " << hist[0][i] << endl;
			}
		}
		break;
	}
	case 3: {
		if (argc < 4) {
			cout << "loi tham so" << endl;
			return -1;
		}
		image = imread(argv[1], IMREAD_ANYCOLOR);
		image1 = imread(argv[2], IMREAD_ANYCOLOR);
		if (!image.data || !image1.data) {
			cout << "khong mo duoc anh " << endl;
			return -1;
		}
		double res = compareHist(image, image1);
		cout << "ket qua so sanh 2 anh : " << res << endl;
		break;
	}
	case 4: {
		if (argc < 6) {
			cout << "loi tham so" << endl;
			return -1;
		}
		image = imread(argv[1], IMREAD_ANYCOLOR);
		if (!image.data) {
			cout << "khong mo duoc anh" << endl;
			return -1;
		}
		if (isGrayScale(image)) {
			cout << "khong phai anh mau" << endl;
			return -1;
		}

		int rBin, gBin, bBin;
		stringstream ss(argv[3]); ss >> rBin;
		stringstream ss1(argv[4]); ss1 >> gBin;
		stringstream ss2(argv[5]); ss2 >> bBin;

		int hist[3][256];
		memset(hist, 0, sizeof(int) * 3 * 256);
		calcHistogram(image, rBin, gBin, bBin, hist);
		cout << "histogram cua anh luong hoa mau : " << endl;

		cout << "red chanel : " << endl;
		for (int i = 0; i < rBin; ++i) {
			cout << "level : " << i << " : " << hist[0][i] << endl;
		}
		cout << "blue chanel : " << endl;
		for (int i = 0; i < bBin; ++i) {
			cout << "level : " << i << " : " << hist[2][i] << endl;
		}
		cout << "grean chanel : " << endl;
		for (int i = 0; i < gBin; ++i) {
			cout << "level : " << i << " : " << hist[1][i] << endl;
		}
		break;
	}
	case 5: {
		if (argc < 4) {
			cout << "loi tham so" << endl;
			return -1;
		}
		image = imread(argv[1], IMREAD_ANYCOLOR);
		if (!image.data) {
			cout << "khong mo duoc anh" << endl;
			return -1;
		}
		if (!isGrayScale(image)) {
			cout << "khong phai anh xam" << endl;
			return -1;
		}

		stringstream ss(argv[3]);
		int gBin; ss >> gBin;

		int grayHist[256]; memset(grayHist, 0, sizeof(int) * 256);
		calcHistogram(image, gBin, grayHist);
		cout << "histogram luong hoa anh xam : " << endl;
		for (int i = 0; i < gBin; ++i) {
			cout << "bin : " << i << " : " << grayHist[i] << endl;
		}

		break;
	}
	case 6: {
		if (argc < 6) {
			cout << "loi tham so" << endl;
			return -1;
		}
		image = imread(argv[1], IMREAD_ANYCOLOR);
		image2 = imread(argv[2], IMREAD_ANYCOLOR);

		if (!image.data || !image2.data) {
			cout << "khong mo duoc anh" << endl;
			return -1;
		}

		int rBin, gBin, bBin;
		stringstream ss(argv[4]); ss >> rBin;
		stringstream ss1(argv[5]); ss1 >> gBin;
		stringstream ss2(argv[6]); ss2 >> bBin;

		double res = compareHist(image1, image2, rBin, gBin, rBin);
		cout << "ket qua so sanh hai anh luong hoa mau : " << res << endl;
		break;
	}

	case 7: {
		if (argc < 5) {
			cout << "loi tham so" << endl;
			return -1;
		}
		image = imread(argv[1], IMREAD_ANYCOLOR);
		image2 = imread(argv[2], IMREAD_ANYCOLOR);

		int gBin;
		stringstream ss(argv[4]); ss >> gBin;
		if (!image.data || !image2.data) {
			cout << "khong mo duoc anh" << endl;
			return -1;
		}

		if (!isGrayScale(image) || !isGrayScale(image2)) {
			cout << "khong phai anh xam" << endl;
			return -1;
		}
		double res = compareHist_g(image, image2, gBin);
		cout << "ket qua so sanh hai luong hoa xam : " << res << endl;
		break;
	}
	case 8: {

		if (argc < 4) {
			cout << "loi tham so" << endl;
			return -1;
		}
		image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

		if (!image.data ) {
			cout << "khong mo duoc anh" << endl;
			return -1;
		}

		double k;
		stringstream ss(argv[3]); ss >> k;
		Mat dst = xGradient(image, k, 1 / (2 + k));
		namedWindow("Display window", WINDOW_AUTOSIZE);
		imshow("Display window", dst);

		break;
	}
	case 9: {
		if (argc < 4) {
			cout << "loi tham so" << endl;
			return -1;
		}
		image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

		if (!image.data) {
			cout << "khong mo duoc anh" << endl;
			return -1;
		}

		double k;
		stringstream ss(argv[3]); ss >> k;
		Mat dst = yGradient(image, k, 1 / (2 + k));
		namedWindow("Display window", WINDOW_AUTOSIZE);
		imshow("Display window", dst);
		break;
	}
	case 10: {
		if (argc < 4) {
			cout << "loi tham so" << endl;
			return -1;
		}
		image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

		if (!image.data) {
			cout << "khong mo duoc anh" << endl;
			return -1;
		}

		double k;
		stringstream ss(argv[3]); ss >> k;
		Mat dst = magnitude(image, k, 1 / (2 + k));
		namedWindow("Display window", WINDOW_AUTOSIZE);
		imshow("Display window", dst);
		break;
	}
	default:
		break;
	}

	waitKey(0);
	return 0;
}
