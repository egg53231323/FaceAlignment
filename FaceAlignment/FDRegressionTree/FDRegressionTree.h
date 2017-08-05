#ifndef FDRegressionTree_H
#define FDRegressionTree_H

#include "FDCVInclude.h"

class FDTrainData;

class FDRegressionNode
{
public:
	//data
	int mChildrenNodesId[2];
	int mIdx[2];
	int mDepth;
	double mThreshold;
	int mLeafNodeId;
	cv::Mat_<double> mValue;

	FDRegressionNode();
	void Print();
	void Read(std::ifstream& fs);
	void Write(std::ofstream& fs);
};

class FDNodeSplitFeature;
class FDRegressionTree
{
public:
	// depth of the tree
	int mMaxDepth;
	// number of maximum nodes
	int mMaxNodesNum;
	int mLeafNodeNum;
	double mNu;
	double mLambda;
	// tree nodes
	std::vector<FDRegressionNode> mVecNodes;

public:
	FDRegressionTree();
	virtual~FDRegressionTree();
	void SetParam(int maxDepth, int featureGenerateCount, double nu, double lambda);
	void Train(FDTrainData &trainData, const std::vector<int> vecSampleIndex, const std::vector<cv::Point2d> &points, const std::vector<std::vector<uchar> > &samplePointPixelValue);

	void Read(std::ifstream& fs);
	void Write(std::ofstream& fs);

protected:
	void GenerateTestFeature(const std::vector<cv::Point2d> &points, std::vector<FDNodeSplitFeature> &features, int generateNum, double lambda);
	void SplitNode(const FDTrainData &trainData, const std::vector<int> &vecSampleIndex, 
		const cv::Mat_<double> &diffSum, const std::vector<cv::Point2d> &points, const std::vector<std::vector<uchar> > &samplePointPixelValue,
		FDNodeSplitFeature &splitFeature, std::vector<int> &leftSampleIndex, std::vector<int> &rightSampleIndex, 
		cv::Mat_<double> &leftDiffSum, cv::Mat_<double> &rightDiffSum);
	
protected:
	int mFeatureGenerateCount;
};

#endif
