#include "FDRegressionTreesModel.h"
#include "FDUtility.h"

FDRegressionTreesModelParam::FDRegressionTreesModelParam()
{
	mStageNum = 10;
	mTreeDepth = 4;
	mTreeNumPerStage = 500;
	mNu = 0.1;
	mLandmarkNum = 68;
	mShapeGenerateNumPerSample = 20; // todo check it
	mLambda = 0.1;
	mFeaturePoolSize = 400;
}

FDRegressionTreesModel::FDRegressionTreesModel()
{

}

FDRegressionTreesModel::~FDRegressionTreesModel()
{

}

void FDRegressionTreesModel::Train(const FDRegressionTreesModelParam &param, FDTrainData &trainData)
{
	mStageRandomPoint.clear();
	mStageRandomPoint.resize(param.mStageNum);

	double minx = 0, miny = 0, maxx = 0, maxy = 0;

	// todo check the minMaxLoc will or not modify the input matrix
	cv::minMaxLoc(trainData.mMeanShape.col(0), &minx, &miny);
	cv::minMaxLoc(trainData.mMeanShape.col(1), &miny, &maxy);
	for (int i = 0; i < param.mStageNum; i++)
	{
		GenerateRandomPoint(mStageRandomPoint[i], param.mFeaturePoolSize, minx, miny, maxx, maxy);
	}

	for (int i = 0; i < param.mStageNum; i++)
	{
		std::vector<int> vecIndex;
		std::vector<cv::Point2d> vecDelta;

		// all stage use meanshape?
		CalcDelta(trainData.mMeanShape, mStageRandomPoint[i], vecIndex, vecDelta);

		int sampleCount = (int)trainData.mVecDataItems.size();
		std::vector<std::vector<uchar> > samplePointPixelValue(sampleCount);
		for (int j = 0; j < sampleCount; j++)
		{
			GetPixelValue(trainData.mVecDataItems[j], trainData.mMeanShape, vecIndex, vecDelta, samplePointPixelValue[j]);
		}
	}
}

void FDRegressionTreesModel::GenerateRandomPoint(std::vector<cv::Point2d> &vecPoint, int count, double minx, double miny, double maxx, double maxy)
{
	cv::RNG rng(FDUtility::GetUInt64Value());
	vecPoint.clear();
	double x = 0, y = 0;
	double ds = 0.0, de = 1.0;
	double deltaX = maxx - minx;
	double deltaY = maxy - miny;
	for (int i = 0; i < count; i++)
	{
		x = rng.uniform(ds, de) * deltaX + minx;
		y = rng.uniform(ds, de) * deltaY + miny;
		vecPoint.push_back(cv::Point2d(x, y));
	}
}

void FDRegressionTreesModel::CalcDelta(const cv::Mat_<double> &shape, const std::vector<cv::Point2d> &vecPoint, 
	std::vector<int> &vecIndex, std::vector<cv::Point2d> &vecDelta)
{
	vecIndex.resize(vecPoint.size());
	vecDelta.resize(vecPoint.size());
	int count = (int)vecPoint.size();
	for (int i = 0; i < count; i++)
	{
		int index = NearestPointIndex(shape, vecPoint[i].x, vecPoint[i].y);
		vecIndex[i] = index;
		vecDelta[i] = vecPoint[i] - cv::Point2d(shape(index, 0), shape(index, 1));
	}
}

int FDRegressionTreesModel::NearestPointIndex(const cv::Mat_<double> &shape, double x, double y)
{
	int count = shape.rows;
	double mindis = std::numeric_limits<double>::max();
	double dis = 0;
	int index = -1;
	for (int i = 0; i < count; i++)
	{
		dis = (shape(i, 0) - x) * (shape(i, 0) - x) + (shape(i, 1) - y) * (shape(i, 1) - y);
		if (dis < mindis)
		{
			mindis = dis;
			index = i;
		}
	}
	return index;
}

void FDRegressionTreesModel::GetPixelValue(const FDTrainDataItem &item,
	const cv::Mat_<double> &referenceShape,
	const std::vector<int> &vecIndex,
	const std::vector<cv::Point2d> &vecDelta,
	std::vector<uchar> &pixelValue)
{
	pixelValue.resize(vecIndex.size());
	cv::Mat_<double> rotation;
	double scale = 0;

	FDUtility::SimilarityTransform(FDUtility::RealToRelative(item.mCurrentShape, item.mBoundingBox), referenceShape, rotation, scale);

	int count = (int)vecIndex.size();
	for (int i = 0; i < count; i++)
	{
		double px = (rotation(0, 0) * vecDelta[i].x + rotation(0, 1) * vecDelta[i].y) * scale;
		double py = (rotation(1, 0) * vecDelta[i].x + rotation(1, 1) * vecDelta[i].y) * scale;
		px = px * item.mBoundingBox.m_width / 2.0;;
		py = py * item.mBoundingBox.m_height / 2.0;
		int x = (int)(px + item.mCurrentShape(vecIndex[i], 0));
		int y = (int)(py + item.mCurrentShape(vecIndex[i], 1));
		if (x < 0 || y < 0 || x >= item.mImage.cols || y >= item.mImage.rows)
		{
			pixelValue[i] = 0;
		}
		else
		{
			pixelValue[i] = item.mImage(y, x);
		}
	}
}
