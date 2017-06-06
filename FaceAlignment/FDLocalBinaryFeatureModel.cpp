#include "stdafx.h"
#include "FDLocalBinaryFeatureModel.h"
#include "FDLocalBinaryFeatureModelImp.h"
#include "FDRandomForest.h"
#include "FDUtility.h"

FDLocalBinaryFeatureModelParam::FDLocalBinaryFeatureModelParam()
{
	mLandmarkNum = 0;
	const std::vector<int> &landmarkFlag = FDUtility::GetLandmarkFlag();
	int landmarkCount = (int)landmarkFlag.size();
	for (int i = 0; i < landmarkCount; i++)
	{
		if (landmarkFlag[i] != 0)
		{
			mLandmarkNum++;
		}
	}
	mShapeGenerateNumPerSample = 5;
	mSampleOverlapRate = 0.3;
	mMaxTreeNum = 15;
	mMaxTreeDepth = 5;
	mStageNum = 5;

	//int count[Stage_Array_Size] = { 500, 500, 500, 300, 300, 200, 200,200,100, 100 };
	int count[Stage_Array_Size] = { 500, 500, 500, 500, 500, 500, 500, 500, 500, 500 };
	//double radius[Stage_Array_Size] = { 0.4, 0.3, 0.2, 0.15, 0.12, 0.10, 0.08, 0.06, 0.06, 0.05 };
	//double radius[Stage_Array_Size] = { 0.29, 0.21, 0.16, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01 };
	//double radius[Stage_Array_Size] = { 0.3, 0.25, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02, 0.01 };
	double radius[Stage_Array_Size] = { 0.25, 0.20, 0.15, 0.10, 0.05, 0.08, 0.06, 0.04, 0.02, 0.01 };
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

bool FDLocalBinaryFeatureModel::Predict(const cv::Mat_<uchar> &image, std::vector<cv::Mat_<double> > &result, std::vector<FDBoundingBox> *pVecBox /*=NULL*/)
{
	return mImp->Predict(image, result, pVecBox);
}

bool FDLocalBinaryFeatureModel::Save(const char *path)
{
	return mImp->Save(path);
}

bool FDLocalBinaryFeatureModel::Load(const char *path)
{
	return mImp->Load(path);
}


