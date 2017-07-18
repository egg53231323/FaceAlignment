#ifndef FDRegressionTree_H
#define FDRegressionTree_H

#include "FDCVInclude.h"

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

	FDRegressionTreesModelParam();
};

class FDBoundingBox;
class FDTrainData;
class FDTrainDataItem;
class FDRegressionTreesModel
{
public:
	FDRegressionTreesModel();
	virtual~FDRegressionTreesModel();

	void Train(const FDRegressionTreesModelParam &param, FDTrainData &trainData);

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
};

#endif