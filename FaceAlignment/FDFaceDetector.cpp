#include "FDFaceDetector.h"
#include "FDUtility.h"
#include "libfacedetection/facedetect-dll.h"

#define DETECT_BUFFER_SIZE 0x20000
#pragma comment(lib,"libfacedetection/libfacedetect-x64.lib")

FDFaceDetector::FDFaceDetector(): mBuffer(NULL)
{
	mBuffer = new unsigned char[DETECT_BUFFER_SIZE];
}

FDFaceDetector::~FDFaceDetector()
{
	if (NULL != mBuffer)
	{
		delete []mBuffer;
	}
}

void FDFaceDetector::SetCascadeClassifierModelPath(const char *path)
{
	mCascadeClassifier.load(path);
}

bool FDFaceDetector::FaceDetect(const cv::Mat_<uchar> &image, std::vector<cv::Rect> &vecFace, int type)
{
	vecFace.clear();
	if (0 == type)
	{
		double scale = 1.3;
		std::vector<cv::Rect> faces;
		cv::Mat smallImg(cvRound(image.rows / scale), cvRound(image.cols / scale), CV_8UC1);
		cv::resize(image, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
		//cv::equalizeHist(smallImg, smallImg);

		mCascadeClassifier.detectMultiScale(smallImg, faces, 1.1, 2, 0
		//|CV_HAAR_FIND_BIGGEST_OBJECT
		//|CV_HAAR_DO_ROUGH_SEARCH
		| CV_HAAR_SCALE_IMAGE,
		cv::Size(30, 30));
		int count = (int)faces.size();
		for (int i = 0; i < count; i++)
		{
			cv::Rect &rc = faces[i];
			rc.x = std::min((int)(rc.x * scale), image.cols);
			rc.y = std::min((int)(rc.y * scale), image.rows);
			rc.width = std::min((int)(rc.width *scale), image.cols - rc.x);
			rc.height = std::min((int)(rc.height * scale), image.rows - rc.y);
			vecFace.push_back(rc);
		}
	}
	else
	{
		int dolandmark = 0;
		int * pResults = facedetect_frontal_surveillance(mBuffer, (unsigned char*)(image.ptr(0)), image.cols, image.rows, (int)image.step, 1.2f, 2, 48, 0, dolandmark);
		int count = (pResults ? *pResults : 0);
		for (int i = 0; i < count; i++)
		{
			short * p = ((short*)(pResults + 1)) + 142 * i;
			int x = p[0];
			int y = p[1];
			int w = p[2];
			int h = p[3];
			int neighbors = p[4];
			int angle = p[5];
			vecFace.push_back(cv::Rect(x, y, w, h));
		}
	}
	return true;
}
