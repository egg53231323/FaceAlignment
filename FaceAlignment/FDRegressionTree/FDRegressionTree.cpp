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
	if (mLeafNodeId >= 0)
	{
		int rows = 0, cols = 0;
		fs.read((char *)&rows, sizeof(int));
		fs.read((char *)&cols, sizeof(int));
		double *data = new double[rows * cols];
		fs.read((char *)data, sizeof(double) * rows * cols);
		mValue = cv::Mat_<double>(rows, cols);
		for (int i = 0; i < mValue.rows; i++)
		{
			for (int j = 0; j < mValue.cols; j++)
			{
				mValue(i, j) = data[i * mValue.cols + j];
			}
		}
		delete[]data;
		data = NULL;
	}
}

void FDRegressionNode::Write(std::ofstream& fs)
{
	fs.write((const char *)mChildrenNodesId, sizeof(int) * 2);
	fs.write((const char *)mIdx, sizeof(int) * 2);
	fs.write((const char *)&mDepth, sizeof(int));
	fs.write((const char *)&mLeafNodeId, sizeof(int));
	fs.write((const char *)&mThreshold, sizeof(double));
	if (mLeafNodeId >= 0)
	{
		fs.write((const char *)&(mValue.rows), sizeof(int));
		fs.write((const char *)&(mValue.cols), sizeof(int));
		double *data = new double[mValue.rows*mValue.cols];
		for (int i = 0; i < mValue.rows; i++)
		{
			for (int j = 0; j < mValue.cols; j++)
			{
				data[i * mValue.cols + j] = mValue(i, j);
			}
		}
		fs.write((const char *)data, sizeof(double) * mValue.rows * mValue.cols);
		delete[]data;
		data = NULL;
	}
}


FDRegressionTree::FDRegressionTree()
{
	mMaxDepth = 5;
	mMaxNodesNum = (int)(pow(2, mMaxDepth) - 1);
	mLeafNodeNum = 0;

	mFeatureGenerateCount = 20;

	mVecNodes.resize(mMaxNodesNum);

	mNu = 0;
	mLambda = 0;
}

FDRegressionTree::~FDRegressionTree()
{

}

void FDRegressionTree::SetParam(int maxDepth, int featureGenerateCount, double nu, double lambda)
{
	mMaxDepth = maxDepth;
	mMaxNodesNum = (int)(pow(2, mMaxDepth) - 1);

	mVecNodes.resize(mMaxNodesNum);

	mFeatureGenerateCount = featureGenerateCount;

	mNu = nu;
	mLambda = lambda;
}

void FDRegressionTree::Train(FDTrainData &trainData, const std::vector<int> vecSampleIndex, const std::vector<cv::Point2d> &points, const std::vector<std::vector<uchar> > &samplePointPixelValue)
{
	mLeafNodeNum = 0;
	std::vector<FDTrainDataItem> &vecTrainDataItem = trainData.mVecDataItems;
	int sampleCount = (int)vecSampleIndex.size();
	cv::Mat_<double> tempMat = vecTrainDataItem[vecSampleIndex[0]].mGroundTruthShape;
	cv::Mat_<double> diffSum = cv::Mat_<double>::zeros(tempMat.rows, tempMat.cols);
	for (int i = 0; i < sampleCount; i++)
	{
		FDTrainDataItem &item = vecTrainDataItem[vecSampleIndex[i]];
		// mGroundTruthShape mCurrentShape 是图像坐标，要变换一下
		item.mShapeResidual = item.mGroundTruthShape - item.mCurrentShape;
		item.mShapeResidual = FDUtility::RealToRelative(item.mShapeResidual, item.mBoundingBox);
		diffSum = diffSum + item.mShapeResidual;
	}

	FDRegressionNode &rootNode = mVecNodes[0];
	rootNode.mDepth = 1;
	rootNode.mValue = diffSum / (double)sampleCount * mNu;

	std::vector<std::vector<int> > vecAllNodeSampleIndex;
	vecAllNodeSampleIndex.resize(mMaxNodesNum);
	vecAllNodeSampleIndex[0] = vecSampleIndex;

	std::vector<cv::Mat_<double> > vecDiffSum(mMaxNodesNum);
	vecDiffSum[0] = diffSum;

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
		const std::vector<int> &currentSampleIndex = vecAllNodeSampleIndex[splitNodeId];
		const cv::Mat_<double> &curDiffSum = vecDiffSum[splitNodeId];
		if (currentNode.mDepth == mMaxDepth)
		{
			currentNode.mLeafNodeId = mLeafNodeNum++;
			continue;
		}

		cv::Mat_<double> leftDiffSum, rightDiffSum;
		SplitNode(trainData, currentSampleIndex, curDiffSum, points, samplePointPixelValue, feature, lchildren, rchildren, leftDiffSum, rightDiffSum);
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
		left.mValue = (leftDiffSum / (double)lchildren.size()) * mNu;
		right.mValue = (rightDiffSum / (double)rchildren.size()) * mNu;

		vecAllNodeSampleIndex[newNodeId].swap(lchildren);
		vecAllNodeSampleIndex[newNodeId + 1].swap(rchildren);

		// add child to split queue
		stackNodeToSplit.push(newNodeId + 1);
		stackNodeToSplit.push(newNodeId);

		newNodeId += 2;
	}

	int nodeCount = (int)mVecNodes.size();
	int leafSampleCount = 0;
	for (int i = 0; i < nodeCount; i++)
	{
		FDRegressionNode &currentNode = mVecNodes[i];
		if (currentNode.mLeafNodeId < 0)
			continue;

		const std::vector<int> &currentSampleIndex = vecAllNodeSampleIndex[i];
		int nodeSampleCount = (int)currentSampleIndex.size();
		leafSampleCount += nodeSampleCount;
		for (int j = 0; j < nodeSampleCount; j++)
		{
			FDTrainDataItem &item = vecTrainDataItem[currentSampleIndex[j]];
			item.mCurrentShape += FDUtility::RelativeToReal(currentNode.mValue, item.mBoundingBox);
		}
	}
	FDLog("tree sample count: %d, leaf sample count %d, %s", sampleCount, leafSampleCount);
	if (sampleCount != leafSampleCount)
	{
		FDLog("tree sample count != leaf sample count !!!!!!!!!!!!!!!!!!");
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
			// point 是meanshape 中的坐标系 (-1 -1) -> (1, 1)
			double dis = PT_DIS(points[feature.mIdx1], points[feature.mIdx2]);
			// todo 对应到(0, 0) -> (1, 1)是不是要 *2， 或者调整lambda ?
			dis = dis * 2;
			double acceptProbability = std::exp(-dis / lambda);
			if (!(acceptProbability > randomNumGenerator.uniform((double)0.0, (double)1.0))) {
				break;
			}
		}
		feature.mThreshold = (randomNumGenerator.uniform((double)0.0, (double)1.0) * 256 - 128) / 2.0;
		features.push_back(feature);
	}
}

void FDRegressionTree::SplitNode(const FDTrainData &trainData, const std::vector<int> &vecSampleIndex, 
	const cv::Mat_<double> &diffSum, const std::vector<cv::Point2d> &points, const std::vector<std::vector<uchar> > &samplePointPixelValue, 
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
	GenerateTestFeature(points, features, mFeatureGenerateCount, mLambda);

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
			if (((double)pixelValue[feature.mIdx1] - (double)pixelValue[feature.mIdx2]) > feature.mThreshold)
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
		int rightCount = sampleCount - leftCount;
		cv::Mat_<double> tempRight = diffSum - tempLeft;
		val = 0;
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
		if (((double)pixelValue[resFeature.mIdx1] - (double)pixelValue[resFeature.mIdx2]) > resFeature.mThreshold)
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

void FDRegressionTree::Read(std::ifstream& fs)
{
	fs.read((char *)&mMaxDepth, sizeof(int));
	fs.read((char *)&mMaxNodesNum, sizeof(int));
	fs.read((char *)&mLeafNodeNum, sizeof(int));

	mVecNodes.resize(mMaxNodesNum);
	for (int i = 0; i < mMaxNodesNum; i++)
	{
		mVecNodes[i].Read(fs);
	}
}

void FDRegressionTree::Write(std::ofstream& fs)
{
	fs.write((const char *)&mMaxDepth, sizeof(int));
	fs.write((const char *)&mMaxNodesNum, sizeof(int));
	fs.write((const char *)&mLeafNodeNum, sizeof(int));

	for (int i = 0; i < mMaxNodesNum; i++)
	{
		mVecNodes[i].Write(fs);
	}
}