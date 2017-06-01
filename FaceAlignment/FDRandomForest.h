#ifndef FDRandomForest_H
#define FDRandomForest_H

#include "FDRandomTree.h"

class FDRandomForest
{
public:
	int mId;
	// face land mark num
	int mLandmarkNum;
	// number of tree
	int mMaxTreeNum;
	int mLeafNodeNum;
	// sample ovarlap rate
	double mSampleOverlapRate;
	// trees
	std::vector<std::vector<FDRandomTree> > mVecTree;

public:
	FDRandomForest();
	void SetParam(int maxTreeNum, int maxTreeDepth, int featureGenerateCount, double featureGenerateRadius, double sampleOverlapRate, int landmarkID);
	void Train(const FDTrainData &trainData);

	void Read(std::ifstream& fs);
	void Write(std::ofstream& fs);
};

#endif

