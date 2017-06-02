#ifndef FDFaceDetector_H
#define FDFaceDetector_H

#include "FDCVInclude.h"

class FDFaceDetector
{
protected:
	cv::CascadeClassifier mCascadeClassifier;
	unsigned char * mBuffer;

public:
	FDFaceDetector();
	virtual ~FDFaceDetector();
	void SetCascadeClassifierModelPath(const char *path);
	bool FaceDetect(const cv::Mat_<uchar> &image, std::vector<cv::Rect> &vecFace, int type);
};

#endif // !FDFaceDetector_H


