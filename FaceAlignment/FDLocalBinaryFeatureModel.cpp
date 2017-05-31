#include "stdafx.h"
#include "FDLocalBinaryFeatureModel.h"
#include "FDLocalBinaryFeatureModelImp.h"
#include "FDRandomForest.h"
#include "FDUtility.h"

FDLocalBinaryFeatureModelParam::FDLocalBinaryFeatureModelParam()
{
	mLandmarkNum = 68;
	mShapeGenerateNumPerSample = 5;
	mSampleOverlapRate = 0.4;
	mMaxTreeNum = 10;
	mMaxTreeDepth = 5;
	mStageNum = 7;

	int count[Stage_Array_Size] = { 500, 500, 500, 300, 300, 200, 200,200,100, 100 };
	double radius[Stage_Array_Size] = { 0.4, 0.3, 0.2, 0.15, 0.12, 0.10, 0.08, 0.06, 0.06, 0.05 };
	memcpy(mFeatureGenerateCount, count, sizeof(int)*Stage_Array_Size);
	memcpy(mFeatureGenerateRadius, radius, sizeof(double)*Stage_Array_Size);
}

FDLocalBinaryFeatureModel::FDLocalBinaryFeatureModel()
{
	mImp = new FDLocalBinaryFeatureModelImp();
}

FDLocalBinaryFeatureModel::~FDLocalBinaryFeatureModel()
{
	if (NULL != mImp)
	{
		delete mImp;
		mImp = NULL;
	}
}

void FDLocalBinaryFeatureModel::SetCascadeClassifierModelPath(const char *path)
{
	mImp->SetCascadeClassifierModelPath(path);
}

void FDLocalBinaryFeatureModel::Train(const FDLocalBinaryFeatureModelParam &param, FDTrainData &trainData)
{
	mImp->Train(param, trainData);
}

bool FDLocalBinaryFeatureModel::Predict(const cv::Mat_<uchar> &image, std::vector<cv::Mat_<double> > &result)
{
	return mImp->Predict(image, result);
}

bool FDLocalBinaryFeatureModel::Save(const char *path)
{
	return mImp->Save(path);
}

bool FDLocalBinaryFeatureModel::Load(const char *path)
{
	return mImp->Load(path);
}


