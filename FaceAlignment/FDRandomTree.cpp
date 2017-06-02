#include "stdafx.h"
#include "FDRandomTree.h"
#include "FDUtility.h"
#include <fstream>
#include <stack>

FDNode::FDNode()
{
	mChildrenNodesId[0] = 0;
	mChildrenNodesId[1] = 0;
	mDepth = 0;
	mLeafNodeId = -1;
	mThreshold = 0;
	mFeature[0] = 0;
	mFeature[1] = 0;
	mFeature[2] = 0;
	mFeature[3] = 0;
}

void FDNode::Print()
{
	std::cout << "node info: "
		<< "mDepth: " << mDepth 
		<< "mChildrenNodesId[0]: " << mChildrenNodesId[0] 
		<< "mChildrenNodesId[1]: " << mChildrenNodesId[1] 
		<< "mThreshold: "<< mThreshold 
		<< "mLeafNodeId: " << mLeafNodeId
		<< "mFeature[0]: " << mFeature[0] 
		<< "mFeature[1]: " << mFeature[1] 
		<< "mFeature[2]: " << mFeature[2] 
		<< "mFeature[3]: " << mFeature[3] 
		<< std::endl;
}

void FDNode::Read(std::ifstream& fs)
{
	fs.read((char *)mChildrenNodesId, sizeof(int) * 2);
	fs.read((char *)&mDepth, sizeof(int));
	fs.read((char *)&mLeafNodeId, sizeof(int));
	fs.read((char *)&mThreshold, sizeof(double));
	fs.read((char *)mFeature, sizeof(double) * 4);
}

void FDNode::Write(std::ofstream& fs)
{
	fs.write((const char *)mChildrenNodesId, sizeof(int) * 2);
	fs.write((const char *)&mDepth, sizeof(int));
	fs.write((const char *)&mLeafNodeId, sizeof(int));
	fs.write((const char *)&mThreshold, sizeof(double));
	fs.write((const char *)mFeature, sizeof(double)*4);
}

FDRandomTree::FDRandomTree()
{
	mLandmarkID = 0;
	mMaxDepth = 5;
	mMaxNodesNum = (int)(pow(2, mMaxDepth) - 1);
	mLeafNodeNum = 0;

	mFeatureGenerateCount = 500;
	mFeatureGenerateRadius = 0.4;
	mSampleOverlapRate = 0.4;

	mVecNodes.resize(mMaxNodesNum);
}

void FDRandomTree::SetParam(int maxDepth, int featureGenerateCount, double featureGenerateRadius, double sampleOverlapRate, int landmarkID)
{
	mMaxDepth = maxDepth;
	mMaxNodesNum = (int)(pow(2, mMaxDepth) - 1);

	mFeatureGenerateCount = featureGenerateCount;
	mFeatureGenerateRadius = featureGenerateRadius;
	mSampleOverlapRate = sampleOverlapRate;

	mLandmarkID = landmarkID;

	mVecNodes.resize(mMaxNodesNum);

	cv::RNG randomNumGenerator(FDUtility::GetUInt64Value());
	cv::Mat_<double> randomPointPairs(mFeatureGenerateCount, 4);
	for (int i = 0; i < mFeatureGenerateCount; i++)
	{
		double x1 = randomNumGenerator.uniform(-1.0, 1.0);
		double y1 = randomNumGenerator.uniform(-1.0, 1.0);
		double x2 = randomNumGenerator.uniform(-1.0, 1.0);
		double y2 = randomNumGenerator.uniform(-1.0, 1.0);
		if ((x1*x1 + y1*y1 > 1.0) || (x2*x2 + y2*y2 > 1.0))
		{
			i--;
			continue;
		}
		if (landmarkID < 8 && x1 < 0 && x2 < 0)
		{
			FDLog("skip");
			i--;
			continue;
		}
		else if (landmarkID > 8 && landmarkID < 17 && x1 > 0 && x2 > 0)
		{
			FDLog("skip");
			i--;
			continue;
		}

		randomPointPairs(i, 0) = x1 * mFeatureGenerateRadius;
		randomPointPairs(i, 1) = y1 * mFeatureGenerateRadius;
		randomPointPairs(i, 2) = x2 * mFeatureGenerateRadius;
		randomPointPairs(i, 3) = y2 * mFeatureGenerateRadius;
	}
	mRandomPointPairs = randomPointPairs;
}

void FDRandomTree::Train(const FDTrainData &trainData, const std::vector<int> vecSampleIndex)
{
	mLeafNodeNum = 0;
	const std::vector<FDTrainDataItem> &vecTrainDataItem = trainData.mVecDataItems;
	int sampleCount = (int)vecSampleIndex.size();
	cv::Mat_<double> shapeResidual(sampleCount, 2);
	//
	for (int i = 0; i < sampleCount; i++)
	{
		const cv::Mat_<double> &src = vecTrainDataItem[vecSampleIndex[i]].mShapeResidual;
		shapeResidual(i, 0) = src(mLandmarkID, 0);
		shapeResidual(i, 1) = src(mLandmarkID, 1);
	}

	FDNode &rootNode = mVecNodes[0];
	rootNode.mDepth = 1;

	std::vector<std::vector<int> > vecAllNodeSampleIndex;
	vecAllNodeSampleIndex.resize(mMaxNodesNum);
	vecAllNodeSampleIndex[0] = vecSampleIndex;

	int newNodeId = 1;
	double threshold = 0;
	double nodeFeature[4] = { 0 };

	std::vector<int> lchildren, rchildren;
	lchildren.reserve(vecSampleIndex.size());
	rchildren.reserve(vecSampleIndex.size());

	std::stack<int> stackNodeToSplit;
	stackNodeToSplit.push(0);
	int splitNodeId = 0;
	while (!stackNodeToSplit.empty())
	{
		splitNodeId = stackNodeToSplit.top();
		stackNodeToSplit.pop();

		FDNode &currentNode = mVecNodes[splitNodeId];
		std::vector<int> &currentSampleIndex = vecAllNodeSampleIndex[splitNodeId];
		if (currentNode.mDepth == mMaxDepth)
		{
			currentNode.mLeafNodeId = mLeafNodeNum++;
			continue;
		}

		SplitNode(trainData, currentSampleIndex, shapeResidual, threshold, nodeFeature, lchildren, rchildren);
		if (lchildren.empty() || rchildren.empty())
		{
			currentNode.mLeafNodeId = mLeafNodeNum++;
			continue;
		}
	
		// update current node feature and child id
		currentNode.mChildrenNodesId[0] = newNodeId;
		currentNode.mChildrenNodesId[1] = newNodeId + 1;
		currentNode.mThreshold = threshold;
		currentNode.mFeature[0] = nodeFeature[0];
		currentNode.mFeature[1] = nodeFeature[1];
		currentNode.mFeature[2] = nodeFeature[2];
		currentNode.mFeature[3] = nodeFeature[3];

		// update child
		FDNode &left = mVecNodes[newNodeId];
		FDNode &right = mVecNodes[newNodeId + 1];

		left.mDepth = currentNode.mDepth + 1;
		right.mDepth = currentNode.mDepth + 1;

		vecAllNodeSampleIndex[newNodeId].swap(lchildren);
		vecAllNodeSampleIndex[newNodeId + 1].swap(rchildren);

		// add child to split queue
		stackNodeToSplit.push(newNodeId + 1);
		stackNodeToSplit.push(newNodeId);

		newNodeId += 2;
	}
}

void FDRandomTree::SplitNode(const FDTrainData &trainData, const std::vector<int> &vecSampleIndex, cv::Mat_<double> &shapeResidual, 
	double& threshold, double* nodeFeature, std::vector<int>& leftSampleIndex, std::vector<int>& rightSampleIndex)
{
	if (vecSampleIndex.empty())
	{
		threshold = 0;
		nodeFeature[0] = nodeFeature[1] = nodeFeature[2] = nodeFeature[3] = 0;
		leftSampleIndex.clear();
		rightSampleIndex.clear();
		return;
	}

	cv::RNG randomNumGenerator(FDUtility::GetUInt64Value());
	cv::Mat_<double> &randomPointPairs = mRandomPointPairs;
	int sampleCount = (int)vecSampleIndex.size();
	cv::Mat_<int> densities(mFeatureGenerateCount, sampleCount);
	for (int i = 0; i < sampleCount; i++)
	{
		const FDTrainDataItem &item = trainData.mVecDataItems[vecSampleIndex[i]];

		cv::Mat_<double> temp = FDUtility::RealToRelative(item.mCurrentShape, item.mBoundingBox);
	
		cv::Mat_<double> rotation;
		double scale = 0;
		FDUtility::SimilarityTransform(temp, trainData.mMeanShape, rotation, scale);
		
		// whether transpose or not ----------
		for (int j = 0; j < mFeatureGenerateCount; j++)
		{
			double tempX1 = rotation(0, 0) * randomPointPairs(j, 0) + rotation(0, 1) * randomPointPairs(j, 1);
			double tempY1 = rotation(1, 0) * randomPointPairs(j, 0) + rotation(1, 1) * randomPointPairs(j, 1);
			tempX1 = tempX1 * scale * item.mBoundingBox.m_width / 2.0;
			tempY1 = tempY1 * scale * item.mBoundingBox.m_height / 2.0;
			int realX1 = (int)(tempX1 + item.mCurrentShape(mLandmarkID, 0));
			int realY1 = (int)(tempY1 + item.mCurrentShape(mLandmarkID, 1));
			realX1 = std::max(0, std::min(realX1, item.mImage.cols - 1));
			realY1 = std::max(0, std::min(realY1, item.mImage.rows - 1));

			double tempX2 = rotation(0, 0) * randomPointPairs(j, 2) + rotation(0, 1) * randomPointPairs(j, 3);
			double tempY2 = rotation(1, 0) * randomPointPairs(j, 2) + rotation(1, 1) * randomPointPairs(j, 3);
			tempX2 = tempX2 * scale * item.mBoundingBox.m_width / 2.0;
			tempY2 = tempY2 * scale * item.mBoundingBox.m_height / 2.0;
			int realX2 = (int)(tempX2 + item.mCurrentShape(mLandmarkID, 0));
			int realY2 = (int)(tempY2 + item.mCurrentShape(mLandmarkID, 1));
			realX2 = std::max(0, std::min(realX2, item.mImage.cols - 1));
			realY2 = std::max(0, std::min(realY2, item.mImage.rows - 1));

			densities(j, i) = ((int)item.mImage(realY1, realX1)) - ((int)item.mImage(realY2, realX2));
		}
	}

	cv::Mat_<int> densitiesSorted = densities.clone();
	cv::sort(densities, densitiesSorted, CV_SORT_ASCENDING);
	std::vector<double> lc1, lc2;
	std::vector<double> rc1, rc2;
	lc1.reserve(sampleCount);
	lc2.reserve(sampleCount);
	rc1.reserve(sampleCount);
	rc2.reserve(sampleCount);

#define VarianceFactor(x) (x)

	double varOverall = (CalcVariance(shapeResidual.col(0)) + CalcVariance(shapeResidual.col(1))) * VarianceFactor(sampleCount);
	double varLeft = 0;
	double varRight = 0;
	double varReduce = 0;
	double maxVarReduce = std::numeric_limits<double>::min();
	double tempThreshold = 0;
	int maxVarReduceId = -1;
	for (int i = 0; i < mFeatureGenerateCount; i++)
	{
		lc1.clear();
		lc2.clear();
		rc1.clear();
		rc2.clear();
		int thresholdIndex = (int)(sampleCount * randomNumGenerator.uniform(0.05, 0.95));
		tempThreshold = densitiesSorted(i, thresholdIndex);

		for (int j = 0; j < sampleCount; j++)
		{
			if (densities(i, j) < tempThreshold)
			{
				lc1.push_back(shapeResidual(j, 0));
				lc2.push_back(shapeResidual(j, 1));
			}
			else
			{
				rc1.push_back(shapeResidual(j, 0));
				rc2.push_back(shapeResidual(j, 1));
			}
		}

		if (lc1.empty() || rc1.empty())
			continue;

		varLeft = (CalcVariance(lc1) + CalcVariance(lc2)) * VarianceFactor(lc1.size());
		varRight = (CalcVariance(rc1) + CalcVariance(rc2)) * VarianceFactor(rc1.size());
		varReduce = varOverall - varLeft - varRight;
		if (varReduce > maxVarReduce)
		{
			maxVarReduce = varReduce;
			threshold = tempThreshold;
			maxVarReduceId = i;
		}
	}

	leftSampleIndex.clear();
	rightSampleIndex.clear();
	if (maxVarReduceId < 0)
		return;

	nodeFeature[0] = randomPointPairs(maxVarReduceId, 0);
	nodeFeature[1] = randomPointPairs(maxVarReduceId, 1);
	nodeFeature[2] = randomPointPairs(maxVarReduceId, 2);
	nodeFeature[3] = randomPointPairs(maxVarReduceId, 3);

	for (int i = 0; i < sampleCount; i++)
	{
		if (densities(maxVarReduceId, i) < threshold)
		{
			leftSampleIndex.push_back(vecSampleIndex[i]);
		}
		else
		{
			rightSampleIndex.push_back(vecSampleIndex[i]);
		}
	}
}

void FDRandomTree::Read(std::ifstream& fs)
{
	fs.read((char *)&mLandmarkID, sizeof(int));
	fs.read((char *)&mMaxDepth, sizeof(int));
	fs.read((char *)&mMaxNodesNum, sizeof(int));
	fs.read((char *)&mLeafNodeNum, sizeof(int));

	mVecNodes.resize(mMaxNodesNum);
	for (int i = 0; i < mMaxNodesNum; i++)
	{
		mVecNodes[i].Read(fs);
	}
}

void FDRandomTree::Write(std::ofstream& fs)
{
	fs.write((const char *)&mLandmarkID, sizeof(int));
	fs.write((const char *)&mMaxDepth, sizeof(int));
	fs.write((const char *)&mMaxNodesNum, sizeof(int));
	fs.write((const char *)&mLeafNodeNum, sizeof(int));

	for (int i = 0; i < mMaxNodesNum; i++)
	{
		mVecNodes[i].Write(fs);
	}
}

double FDRandomTree::CalcVariance(const std::vector<double>& v)
{
	if (v.empty())
		return 0;

	cv::Mat_<double> mat(v);
	return CalcVariance(mat);
}
double FDRandomTree::CalcVariance(const cv::Mat_<double>& mat)
{
	cv::Mat mean, stddev;
	cv::meanStdDev(mat, mean, stddev);
	return stddev.at<double>(0, 0) * stddev.at<double>(0, 0);
}


