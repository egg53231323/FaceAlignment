#include "FDRegressionTreesModel.h"
#include "FDUtility.h"
#include "FDRegressionTree.h"
#include <fstream>

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
	mFeatureGenerateCount = 20;
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
	mVecIndex.clear();
	mVecIndex.resize(param.mStageNum);
	mVecDelta.clear();
	mVecDelta.resize(param.mStageNum);
	mForests.clear();
	mForests.resize(param.mStageNum);

	double minx = 0, miny = 0, maxx = 0, maxy = 0;

	// todo check the minMaxLoc will or not modify the input matrix
	cv::minMaxLoc(trainData.mMeanShape.col(0), &minx, &miny);
	cv::minMaxLoc(trainData.mMeanShape.col(1), &miny, &maxy);
	for (int i = 0; i < param.mStageNum; i++)
	{
		GenerateRandomPoint(mStageRandomPoint[i], param.mFeaturePoolSize, minx, miny, maxx, maxy);
	}

	std::vector<int> vecSampleIndex;
	int sampleCount = (int)trainData.mVecDataItems.size();
	for (int i = 0; i < sampleCount; i++)
	{
		vecSampleIndex.push_back(i);
	}
	for (int i = 0; i < param.mStageNum; i++)
	{
		CalcDelta(trainData.mMeanShape, mStageRandomPoint[i], mVecIndex[i], mVecDelta[i]);

		int sampleCount = (int)trainData.mVecDataItems.size();
		std::vector<std::vector<uchar> > samplePointPixelValue(sampleCount);
		for (int j = 0; j < sampleCount; j++)
		{
			GetPixelValue(trainData.mVecDataItems[j], trainData.mMeanShape, mVecIndex[i], mVecDelta[i], samplePointPixelValue[j]);
		}
		mForests[i].resize(param.mTreeNumPerStage);
		for (int j = 0; j < param.mTreeNumPerStage; j++)
		{
			mForests[i][j].SetParam(param.mTreeDepth, param.mFeatureGenerateCount, param.mNu, param.mLambda);
			mForests[i][j].Train(trainData, vecSampleIndex, mStageRandomPoint[i], samplePointPixelValue);
		}
	}
}

bool FDRegressionTreesModel::Predict(const cv::Mat_<uchar> &image, std::vector<cv::Mat_<double> > &result, std::vector<FDBoundingBox> *pVecBox /*=NULL*/)
{
	uint64 t1 = FDUtility::GetCurrentTime();

	std::vector<cv::Rect> faces;
	mFaceDetector.FaceDetect(image, faces, mFaceDetectorType);

	uint64 t2 = FDUtility::GetCurrentTime();
	FDLog("cost detect %d", (int)(t2 - t1));
	if (faces.empty())
		return false;

	FDBoundingBox boundingBox;

	int faceCount = (int)faces.size();
	result.resize(faceCount);
	for (int i = 0; i < faceCount; i++)
	{
		FDUtility::RectToBoundingBox(faces[i], boundingBox);

		Predict(image, result[i], boundingBox);
		if (NULL != pVecBox)
		{
			pVecBox->push_back(boundingBox);
		}
	}
	return true;
}

bool FDRegressionTreesModel::Predict(const cv::Mat_<uchar> &image, cv::Mat_<double> &result, const FDBoundingBox &boudingBox)
{
	uint64 t3 = FDUtility::GetCurrentTime();

	int stageNum = (int)mForests.size();
	mPredictData.mVecDataItems.clear();
	mPredictData.mVecDataItems.push_back(FDTrainDataItem());
	FDTrainDataItem &item = mPredictData.mVecDataItems[0];
	item.mBoundingBox = boudingBox;
	item.mImage = image;
	item.mCurrentShape = FDUtility::RelativeToReal(mPredictData.mMeanShape, item.mBoundingBox);
	for (int i = 0; i < stageNum; i++)
	{
		std::vector<uchar> samplePointPixelValue;
		GetPixelValue(item, mPredictData.mMeanShape, mVecIndex[i], mVecDelta[i], samplePointPixelValue);
		int treeCount = (int)mForests[i].size();
		for (int j = 0; j < treeCount; j++)
		{
			FDRegressionTree &tree = mForests[i][j];
			int nodeId = 0;
			while (true)
			{
				FDRegressionNode &node = tree.mVecNodes[nodeId];
				if (node.mLeafNodeId >= 0)
				{
					item.mCurrentShape += node.mValue;
					break;
				}
				if ((samplePointPixelValue[node.mIdx[0]] - samplePointPixelValue[node.mIdx[1]]) > node.mThreshold)
				{
					nodeId = node.mChildrenNodesId[0];
				}
				else
				{
					nodeId = node.mChildrenNodesId[1];
				}
			}
			

		}
	}


	result = item.mCurrentShape;
	mPredictData.mVecDataItems.clear();

	uint64 t4 = FDUtility::GetCurrentTime();
	FDLog("cost predict %d", (int)(t4 - t3));
	return true;
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

bool FDRegressionTreesModel::Save(const char *path)
{
	if (mForests.empty() || mStageRandomPoint.size() != mForests.size())
		return false;

	std::ofstream fs(path, std::ios::binary);

	int stageNum = (int)mStageRandomPoint.size();
	fs.write((const char *)&stageNum, sizeof(int));
	for (int i = 0; i < stageNum; i++)
	{
		std::vector<cv::Point2d> &points = mStageRandomPoint[i];
		int count = (int)points.size();
		fs.write((const char *)&count, sizeof(int));
		for (int j = 0; j < count; j++)
		{
			fs.write((const char *)&(points[j].x), sizeof(double));
			fs.write((const char *)&(points[j].y), sizeof(double));
		}
	}

	for (int i = 0; i < stageNum; i++)
	{
		std::vector<FDRegressionTree> &trees = mForests[i];
		int count = (int)trees.size();
		fs.write((const char *)&count, sizeof(int));
		for (int j = 0; j < count; j++)
		{
			trees[j].Write(fs);
		}
	}
	WriteMeanShape(fs);
	fs.write((const char *)&mFaceDetectorType, sizeof(int));
	return true;
}

bool FDRegressionTreesModel::Load(const char *path)
{
	std::ifstream fs(path, std::ios::binary);

	int stageNum = 0;
	fs.read((char *)&stageNum, sizeof(int));
	mStageRandomPoint.resize(stageNum);
	for (int i = 0; i < stageNum; i++)
	{
		std::vector<cv::Point2d> &points = mStageRandomPoint[i];
		int count = 0;
		fs.read((char *)&count, sizeof(int));
		points.resize(count);
		for (int j = 0; j < count; j++)
		{
			fs.read((char *)&(points[j].x), sizeof(double));
			fs.read((char *)&(points[j].y), sizeof(double));
		}
	}

	mForests.resize(stageNum);
	for (int i = 0; i < stageNum; i++)
	{
		std::vector<FDRegressionTree> &trees = mForests[i];
		int count = 0;
		fs.read((char *)&count, sizeof(int));
		trees.resize(count);
		for (int j = 0; j < count; j++)
		{
			trees[j].Read(fs);
		}
	}
	ReadMeanShape(fs);
	fs.read((char *)&mFaceDetectorType, sizeof(int));
	fs.close();
	std::cout << "face detector type: " << mFaceDetectorType << std::endl;

	mVecIndex.resize(stageNum);
	mVecDelta.resize(stageNum);
	for (int i = 0; i < stageNum; i++)
	{
		CalcDelta(mPredictData.mMeanShape, mStageRandomPoint[i], mVecIndex[i], mVecDelta[i]);
	}
	return !mForests.empty();
}

void FDRegressionTreesModel::ReadMeanShape(std::ifstream& fs)
{
	int rows = 0;
	fs.read((char *)&rows, sizeof(int));
	double *val = new double[rows * 2];
	fs.read((char *)val, sizeof(double) * rows * 2);
	mPredictData.mMeanShape = cv::Mat_<double>(rows, 2);
	for (int i = 0; i < rows; i++)
	{
		mPredictData.mMeanShape(i, 0) = val[i * 2];
		mPredictData.mMeanShape(i, 1) = val[i * 2 + 1];
	}
	delete[]val;
	val = NULL;
}

void FDRegressionTreesModel::WriteMeanShape(std::ofstream& fs)
{
	int rows = mPredictData.mMeanShape.rows;
	fs.write((const char *)&rows, sizeof(int));
	double *val = new double[rows * 2];
	for (int i = 0; i < rows; i++)
	{
		val[i * 2] = mPredictData.mMeanShape(i, 0);
		val[i * 2 + 1] = mPredictData.mMeanShape(i, 1);
	}
	fs.write((const char *)val, sizeof(double) * rows * 2);
	delete[]val;
	val = NULL;
}