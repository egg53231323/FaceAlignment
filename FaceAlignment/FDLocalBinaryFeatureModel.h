#ifndef FDLocalBinaryFeatureModel_H
#define FDLocalBinaryFeatureModel_H

#include "FDCVInclude.h"

#define Stage_Array_Size 10

class FDLocalBinaryFeatureModelParam
{
public:
	int mLandmarkNum;
	int mShapeGenerateNumPerSample;
	double mSampleOverlapRate;

	int mMaxTreeNum;
	int mMaxTreeDepth;

	int mStageNum;
	int mFeatureGenerateCount[Stage_Array_Size];
	double mFeatureGenerateRadius[Stage_Array_Size];

	FDLocalBinaryFeatureModelParam();
};

class FDBoundingBox;
class FDTrainData;
class FDLocalBinaryFeatureModelImp;
class FDLocalBinaryFeatureModel
{
public:
	FDLocalBinaryFeatureModel();
	virtual ~FDLocalBinaryFeatureModel();

	void SetCascadeClassifierModelPath(const char *path);
	void Train(const FDLocalBinaryFeatureModelParam &param, FDTrainData &trainData);
	bool Predict(const cv::Mat_<uchar> &image, std::vector<cv::Mat_<double> > &result, std::vector<FDBoundingBox> *pVecBox = NULL);

	bool Save(const char *path);
	bool Load(const char *path);

protected:
	FDLocalBinaryFeatureModelImp *mImp;
};

#endif
