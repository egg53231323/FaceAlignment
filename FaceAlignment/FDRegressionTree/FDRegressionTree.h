#ifndef FDRegressionTree_H
#define FDRegressionTree_H

#include "FDCVInclude.h"

class FDRegressionTreeFeature
{
public:
	FDRegressionTreeFeature() : mIndex1(0), mIndex2(0), mThreshold(0) {}
	int mIndex1;
	int mIndex2;
	double mThreshold;
};

class FDRegressionTree
{
public:
	FDRegressionTree();
	virtual~FDRegressionTree();
	
protected:
	std::vector<FDRegressionTreeFeature> mFeatures;
	std::vector<cv::Mat_<double> > mLeafValues;
};

#endif
