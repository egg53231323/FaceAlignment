#ifndef FDRegressionModel_H
#define FDRegressionModel_H

#include "FDCVInclude.h"
#include "FDUtility.h"
#include "FDFaceDetector.h"

class FDRegressionTreesModelParam
{
public:
	int mStageNum;
	int mTreeNumPerStage;
	int mTreeDepth;
	double mNu;
	int mLandmarkNum;
	int mShapeGenerateNumPerSample;
	double mLambda;
	int mFeaturePoolSize;
	int mFeatureGenerateCount;

	FDRegressionTreesModelParam();
};

class FDBoundingBox;
class FDTrainData;
class FDTrainDataItem;
class FDRegressionTree;
class FDRegressionTreesModel
{
public:
	FDRegressionTreesModel();
	virtual~FDRegressionTreesModel();

	void Train(const FDRegressionTreesModelParam &param, FDTrainData &trainData);
	bool Predict(const cv::Mat_<uchar> &image, std::vector<cv::Mat_<double> > &result, std::vector<FDBoundingBox> *pVecBox = NULL);
	bool Predict(const cv::Mat_<uchar> &image, cv::Mat_<double> &result, const FDBoundingBox &boudingBox);

protected:
	void GenerateRandomPoint(std::vector<cv::Point2d> &vecPoint, int count, double minx, double miny, double maxx, double maxy);

	void CalcDelta(const cv::Mat_<double> &shape, const std::vector<cv::Point2d> &vecPoint, 
		std::vector<int> &vecIndex, std::vector<cv::Point2d> &vecDelta);

	int NearestPointIndex(const cv::Mat_<double> &shape, double x, double y);

	void GetPixelValue(const FDTrainDataItem &item, 
		const cv::Mat_<double> &referenceShape, 
		const std::vector<int> &vecIndex, 
		const std::vector<cv::Point2d> &vecDelta,
		std::vector<uchar> &pixelValue);

protected:
	std::vector<std::vector<cv::Point2d> > mStageRandomPoint;
	std::vector<std::vector<int> > mVecIndex;
	std::vector<std::vector<cv::Point2d> > mVecDelta;
	std::vector<std::vector<FDRegressionTree> > mForests;
	FDTrainData mPredictData;
	FDFaceDetector mFaceDetector;
	int mFaceDetectorType;
};

#endif