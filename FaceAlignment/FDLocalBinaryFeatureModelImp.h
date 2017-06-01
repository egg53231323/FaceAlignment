#ifndef FDLocalBinaryFeatureModelImp_H
#define FDLocalBinaryFeatureModelImp_H

#include <vector>
#include <fstream>
#include "FDRandomForest.h"
#include "FDUtility.h"

struct feature_node;
struct model;
class FDLocalBinaryFeatureModelParam;
class FDLocalBinaryFeatureModelImp
{
protected:
	std::vector<FDRandomForest> mVecRandomForest;
	std::vector<std::vector<struct model*> > mVecModels;
	FDTrainData mPredictData;
	cv::CascadeClassifier mCascadeClassifier;

public:
	virtual ~FDLocalBinaryFeatureModelImp();
	void SetCascadeClassifierModelPath(const char *path);
	void Train(const FDLocalBinaryFeatureModelParam &param, FDTrainData &trainData);
	bool Predict(const cv::Mat_<uchar> &image, std::vector<cv::Mat_<double> > &result, std::vector<FDBoundingBox> *pVecBox =NULL);
	bool Predict(const cv::Mat_<uchar> &image, cv::Mat_<double> &result, const FDBoundingBox &boudingBox);

	bool Save(const char *path);
	bool Load(const char *path);

protected:
	void GetShapeResidual(FDTrainData &trainData);

	feature_node** GenerateFeature(const FDRandomForest &randomForest, FDTrainData &trainData);

	void GetCodeFromRandomForest(feature_node *feature, const FDRandomForest &randomForest,
		const FDTrainDataItem &item, const cv::Mat_<double>& rotation, double scale);

	void GlobalRegression(feature_node **feature, FDTrainData &trainData, std::vector<model*> &vecModel, int leafFeatureNum);
	void GlobalPrediction(feature_node **feature, const std::vector<model*> &vecModel, cv::Mat_<double> &currentShape, const FDBoundingBox &boundingBox);

	void ReleaseFeature(feature_node **features, int num);
	void ReleaseModel();

	void ReadRandomForest(std::ifstream& fs);
	void WriteRandomForest(std::ofstream& fs);

	void ReadRegression(std::ifstream& fs);
	void WriteRegression(std::ofstream& fs);

	void ReadMeanShape(std::ifstream& fs);
	void WriteMeanShape(std::ofstream& fs);

};

#endif
