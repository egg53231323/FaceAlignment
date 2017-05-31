#ifndef FDRandomTree_H
#define FDRandomTree_H

#include <iostream>
#include <vector>
#include "FDCVInclude.h"

class FDTrainData;

class FDNode
{
public:
	//data
	int mChildrenNodesId[2];
	int mDepth;
	double mThreshold;
	double mFeature[4];

	FDNode();
	void Print();
	void Read(std::ifstream& fs);
	void Write(std::ofstream& fs);
};

class FDRandomTree
{
public:
	// id of the landmark
	int mLandmarkID;
	// depth of the tree
	int mMaxDepth;
	// number of maximum nodes
	int mMaxNodesNum;
	// tree nodes
	std::vector<FDNode> mVecNodes;


protected:
	//////////////////////////////////////
	// train temp info, will not serialize
	// pixel featurs number
	int mFeatureGenerateCount;
	// random point generate radius
	double mFeatureGenerateRadius;
	// sample ovarlap rate
	double mSampleOverlapRate;
	//////////////////////////////////////

public:
	FDRandomTree();
	void SetParam(int maxDepth, int featureGenerateCount, double featureGenerateRadius, double sampleOverlapRate, int landmarkID);
	void Train(const FDTrainData &trainData, const std::vector<int> vecSampleIndex);

	void Read(std::ifstream& fs);
	void Write(std::ofstream& fs);

protected:
	//Splite the node
	void SplitNode(const FDTrainData &trainData, const std::vector<int> &vecSampleIndex, cv::Mat_<double> &shapeResidual,
		double& threshold, double* nodeFeature, std::vector<int>& leftSampleIndex, std::vector<int>& rightSampleIndex);

	double CalcVar(const std::vector<double> &v);
	double CalcVar(const cv::Mat_<double> &mat);
};



#endif