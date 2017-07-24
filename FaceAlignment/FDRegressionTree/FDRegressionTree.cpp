#include "FDRegressionTree.h"
#include "FDUtility.h"
#include <fstream>
#include <stack>

class FDNodeSplitFeature
{
public:
	FDNodeSplitFeature() : mIdx1(0), mIdx2(0), mThreshold(0) {}
	int mIdx1;
	int mIdx2;
	double mThreshold;
};

#define PT_DIS(pt1, pt2) sqrt(((pt1).x - (pt2).x) * ((pt1).x - (pt2).x) + ((pt1).y - (pt2).y) * ((pt1).y - (pt2).y))

FDRegressionNode::FDRegressionNode()
{
	mChildrenNodesId[0] = 0;
	mChildrenNodesId[1] = 0;
	mIdx[0] = 0;
	mIdx[1] = 0;
	mDepth = 0;
	mLeafNodeId = -1;
	mThreshold = 0;
}

void FDRegressionNode::Print()
{
	std::cout << "node info: "
		<< "mChildrenNodesId[0]: " << mChildrenNodesId[0]
		<< "mChildrenNodesId[1]: " << mChildrenNodesId[1]
		<< "mIdx[0]: " << mIdx[0]
		<< "mIdx[1]: " << mIdx[1]
		<< "mThreshold: " << mThreshold
		<< std::endl;
}

void FDRegressionNode::Read(std::ifstream& fs)
{
	fs.read((char *)mChildrenNodesId, sizeof(int) * 2);
	fs.read((char *)mIdx, sizeof(int) * 2);
	fs.read((char *)&mDepth, sizeof(int));
	fs.read((char *)&mLeafNodeId, sizeof(int));
	fs.read((char *)&mThreshold, sizeof(double));
}

void FDRegressionNode::Write(std::ofstream& fs)
{
	fs.write((const char *)mChildrenNodesId, sizeof(int) * 2);
	fs.write((const char *)mIdx, sizeof(int) * 2);
	fs.write((const char *)&mDepth, sizeof(int));
	fs.write((const char *)&mLeafNodeId, sizeof(int));
	fs.write((const char *)&mThreshold, sizeof(double));
	fs.write((const char *)&mLeafNodeId, sizeof(int));
}


FDRegressionTree::FDRegressionTree()
{
	mMaxDepth = 5;
	mMaxNodesNum = (int)(pow(2, mMaxDepth) - 1);
	mLeafNodeNum = 0;

	mFeatureGenerateCount = 500;

	mVecNodes.resize(mMaxNodesNum);
}

FDRegressionTree::~FDRegressionTree()
{

}

void FDRegressionTree::Train(FDTrainData &trainData, const std::vector<int> vecSampleIndex, const std::vector<cv::Point2d> &points, const std::vector<std::vector<uchar> > &samplePointPixelValue)
{
	// 求和，下一步分裂

	mLeafNodeNum = 0;
	std::vector<FDTrainDataItem> &vecTrainDataItem = trainData.mVecDataItems;
	int sampleCount = (int)vecSampleIndex.size();
	cv::Mat_<double> tempMat = vecTrainDataItem[vecSampleIndex[0]].mGroundTruthShape;
	cv::Mat_<double> diffSum = cv::Mat_<double>::zeros(tempMat.rows, tempMat.cols);
	for (int i = 0; i < sampleCount; i++)
	{
		FDTrainDataItem &item = vecTrainDataItem[vecSampleIndex[i]];
		item.mShapeResidual = item.mGroundTruthShape - item.mCurrentShape;
		diffSum = diffSum + item.mShapeResidual;
	}

	FDRegressionNode &rootNode = mVecNodes[0];
	rootNode.mDepth = 1;

	std::vector<std::vector<int> > vecAllNodeSampleIndex;
	vecAllNodeSampleIndex.resize(mMaxNodesNum);
	vecAllNodeSampleIndex[0] = vecSampleIndex;

	int newNodeId = 1;

	std::vector<int> lchildren, rchildren;
	lchildren.reserve(vecSampleIndex.size());
	rchildren.reserve(vecSampleIndex.size());

	std::stack<int> stackNodeToSplit;
	stackNodeToSplit.push(0);
	int splitNodeId = 0;
	FDNodeSplitFeature feature;
	while (!stackNodeToSplit.empty())
	{
		splitNodeId = stackNodeToSplit.top();
		stackNodeToSplit.pop();

		FDRegressionNode &currentNode = mVecNodes[splitNodeId];
		std::vector<int> &currentSampleIndex = vecAllNodeSampleIndex[splitNodeId];
		if (currentNode.mDepth == mMaxDepth)
		{
			currentNode.mLeafNodeId = mLeafNodeNum++;
			continue;
		}

		cv::Mat_<double> leftDiffSum, rightDiffSum;
		SplitNode(trainData, currentSampleIndex, diffSum, points, samplePointPixelValue, feature, lchildren, rchildren, leftDiffSum, rightDiffSum);
		if (lchildren.empty() || rchildren.empty())
		{
			currentNode.mLeafNodeId = mLeafNodeNum++;
			continue;
		}

		// update current node feature and child id
		currentNode.mChildrenNodesId[0] = newNodeId;
		currentNode.mChildrenNodesId[1] = newNodeId + 1;
		currentNode.mThreshold = feature.mThreshold;
		currentNode.mIdx[0] = feature.mIdx1;
		currentNode.mIdx[1] = feature.mIdx2;

		// update child
		FDRegressionNode &left = mVecNodes[newNodeId];
		FDRegressionNode &right = mVecNodes[newNodeId + 1];

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

void FDRegressionTree::GenerateTestFeature(const std::vector<cv::Point2d> &points, std::vector<FDNodeSplitFeature> &features, int generateNum, double lambda)
{
	features.clear();
	FDNodeSplitFeature feature;
	cv::RNG randomNumGenerator(FDUtility::GetUInt64Value());
	int maxIdx = (int)points.size() - 1;
	for (int i = 0; i < generateNum; i++)
	{
		while (true)
		{
			feature.mIdx1 = randomNumGenerator.uniform((int)0, maxIdx);
			feature.mIdx2 = randomNumGenerator.uniform((int)0, maxIdx);
			if (feature.mIdx1 == feature.mIdx2) {
				continue;
			}
			// todo points 是（0， 1）坐标？
			double dis = PT_DIS(points[feature.mIdx1], points[feature.mIdx2]);
			double acceptProbability = std::exp(-dis / lambda);
			if (!(acceptProbability > randomNumGenerator.uniform((double)0, (double)1))) {
				break;
			}
		}
		feature.mThreshold = (randomNumGenerator.uniform((double)0, (double)1) * 256 - 128) / 2.0;
		features.push_back(feature);
	}
}

void FDRegressionTree::SplitNode(const FDTrainData &trainData, const std::vector<int> &vecSampleIndex, 
	cv::Mat_<double> &diffSum, const std::vector<cv::Point2d> &points, const std::vector<std::vector<uchar> > &samplePointPixelValue, 
	FDNodeSplitFeature &splitFeature, std::vector<int> &leftSampleIndex, std::vector<int> &rightSampleIndex, 
	cv::Mat_<double> &leftDiffSum, cv::Mat_<double> &rightDiffSum)
{
	if (vecSampleIndex.empty())
	{
		splitFeature.mThreshold = 0;
		splitFeature.mIdx1 = 0;
		splitFeature.mIdx2 = 0;
		leftSampleIndex.clear();
		rightSampleIndex.clear();
		return;
	}

	std::vector<FDNodeSplitFeature> features;
	// todo 参数设置
	GenerateTestFeature(points, features, mFeatureGenerateCount, 0.5);

	std::vector<cv::Mat_<double> > leftTempSum(mFeatureGenerateCount, cv::Mat_<double>::zeros(diffSum.rows, diffSum.cols));
	std::vector<int> leftTempCount(mFeatureGenerateCount, 0);


	int sampleCount = (int)vecSampleIndex.size();
	cv::Mat_<int> densities(mFeatureGenerateCount, sampleCount);
	for (int i = 0; i < sampleCount; i++)
	{
		const FDTrainDataItem &item = trainData.mVecDataItems[vecSampleIndex[i]];
		const std::vector<uchar> &pixelValue = samplePointPixelValue[vecSampleIndex[i]];
		for (int j = 0; j < mFeatureGenerateCount; j++)
		{
			const FDNodeSplitFeature &feature = features[j];
			if ((pixelValue[feature.mIdx1] - pixelValue[feature.mIdx2]) > feature.mThreshold)
			{
				leftTempSum[j] += item.mShapeResidual;
				leftTempCount[j]++;
			}
		}
	}

	double maxValue = 0, val = 0;
	int maxValueIdx = 0;
	for (int j = 0; j < mFeatureGenerateCount; j++)
	{
		const cv::Mat_<double> &tempLeft = leftTempSum[j];
		int leftCount = leftTempCount[j];
		int rightCount = sampleCount = leftCount;
		cv::Mat_<double> tempRight = diffSum - tempLeft;
		if (leftCount > 0)
		{
			val += tempLeft.dot(tempLeft) / leftCount;
		}
		if (rightCount > 0)
		{
			val += tempRight.dot(tempRight) / rightCount;
		}
		if (val > maxValue)
		{
			maxValue = val;
			maxValueIdx = j;
		}
	}


	const FDNodeSplitFeature &resFeature = features[maxValueIdx];
	leftSampleIndex.clear();
	rightSampleIndex.clear();
	leftDiffSum = cv::Mat_<double>::zeros(diffSum.rows, diffSum.cols);
	for (int i = 0; i < sampleCount; i++)
	{
		const FDTrainDataItem &item = trainData.mVecDataItems[vecSampleIndex[i]];
		const std::vector<uchar> &pixelValue = samplePointPixelValue[vecSampleIndex[i]];
		if ((pixelValue[resFeature.mIdx1] - pixelValue[resFeature.mIdx2]) > resFeature.mThreshold)
		{
			leftDiffSum += item.mShapeResidual;
			leftSampleIndex.push_back(i);
		}
		else
		{
			rightSampleIndex.push_back(i);
		}
	}
	rightDiffSum = diffSum - leftDiffSum;
	splitFeature = resFeature;
}