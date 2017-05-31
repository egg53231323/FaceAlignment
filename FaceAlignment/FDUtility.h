#ifndef FDUTILITY_H
#define FDUTILITY_H

#include "FDCVInclude.h"
#include <string>

#define FDLog(fmt, ...) FDUtility::Log(fmt, __VA_ARGS__)

#define FD_TEMP_DIR "E:/work/test/FaceDetection/test/"

class FDBoundingBox
{
public:
	FDBoundingBox(): m_x(0), m_y(0), m_width(0), m_height(0), m_centerX(0), m_centerY(0){}
	void CalcCenter();

public:
	double m_x;
	double m_y;
	double m_width;
	double m_height;
	double m_centerX;
	double m_centerY;
};

class FDTrainDataItem
{
public:
	cv::Mat_<uchar> mImage;
	cv::Mat_<double> mGroundTruthShape;
	cv::Mat_<double> mCurrentShape;
	cv::Mat_<double> mShapeResidual;
	FDBoundingBox mBoundingBox;
};

class FDTrainData
{
public:
	std::vector<FDTrainDataItem> mVecDataItems;
	cv::Mat_<double> mMeanShape;
};

class FDUtility
{
public:
	static uint64 GetUInt64Value();
	static uint64 GetCurrentTime();
	static void SimilarityTransform(const cv::Mat_<double> &shape1, const cv::Mat_<double> &shape2, cv::Mat_<double> &rotation, double &scale);
	static cv::Mat_<double> RealToRelative(const cv::Mat_<double> &shape, const FDBoundingBox &boundingBox);
	static cv::Mat_<double> RelativeToReal(const cv::Mat_<double> &shape, const FDBoundingBox &boundingBox);

	static void GenerateTrainData(std::vector<std::string> &vecFileListPath, const std::string &cascadeClassifierModelPath, 
		int shapeGenerateNumPerSample, FDTrainData &trainData, std::vector<std::string> *pVecPath = NULL);

	static void CalcMeanShape(FDTrainData &trainData);

	static void OutputItemInfo(FDTrainDataItem &item, const char *path);
	static void DrawShape(cv::Mat_<double> &shape, cv::Mat &img, unsigned char val, int radius = 3);

	static void ShowImage(const char *windowName, const char *path);
	static void Log(const char *fmt, ...);
	static std::string& StdStringFormat(std::string & str, const char * fmt, ...);
	static std::string Replace(const std::string &str, const std::string &strSrc, const std::string &strDst);
	static bool LoadTrainData(const std::string &fileListPath, const std::string &cascadeClassifierModelPath, 
		std::vector<FDTrainDataItem> &vecData, std::vector<std::string> *pVecPath);
	static cv::Mat_<double> LoadGroundTruthShape(const std::string &filePath);
	static void AdjustImage(cv::Mat_<uchar> &image, cv::Mat_<double> &ground_truth_shape, FDBoundingBox &boundingBox);
	static bool IsShapeInRect(const cv::Mat_<double> &shape, const cv::Rect &rect, double scale);
	static bool GetShapeBoundingBox(const cv::Mat_<double> &shape, FDBoundingBox &boundingBox);
	static bool IsBoundingBoxInRect(const FDBoundingBox &boundingBox, const cv::Rect &rect);
	static cv::Rect ScaleRect(const cv::Rect &rect, double scale);
};

#endif // UTILITY_H

