#include "stdafx.h"
#include "FDRandomForest.h"
#include "FDUtility.h"
#include <fstream>

FDRandomForest::FDRandomForest()
{
	mId = 0;
	mLandmarkNum = 68;
	mMaxTreeNum = 10;
	mSampleOverlapRate = 0.4;
	mVecTree.resize(mLandmarkNum);
	for (int i = 0; i < mMaxTreeNum; i++)
	{
		mVecTree[i].resize(mMaxTreeNum);
	}
}

void FDRandomForest::SetParam(int maxTreeNum, int maxTreeDepth, int featureGenerateCount, double featureGenerateRadius, double sampleOverlapRate, int landmarkNum)
{
	mLandmarkNum = landmarkNum;
	mMaxTreeNum = maxTreeNum;
	mSampleOverlapRate = sampleOverlapRate;

	mVecTree.resize(mLandmarkNum);
	for (int i = 0; i < mLandmarkNum; i++)
	{
		mVecTree[i].resize(mMaxTreeNum);
		for (int j = 0; j < mMaxTreeNum; j++)
		{
			mVecTree[i][j].SetParam(maxTreeDepth, featureGenerateCount, featureGenerateRadius, sampleOverlapRate, i);
		}
	}
}

void FDRandomForest::Train(const FDTrainData &trainData)
{
	int sampleCount = (int)trainData.mVecDataItems.size();
	int samplePerTree = (int)((double)(sampleCount) / ((1 - mSampleOverlapRate) * mMaxTreeNum));
	std::vector<int> vecSampleIndex;
	vecSampleIndex.reserve(samplePerTree + 1);
	int indexStart = 0, indexEnd = 0;
	for (int i = 0; i < mLandmarkNum; i++)
	{
		for (int j = 0; j < mMaxTreeNum; j++)
		{
			FDLog("train random tree:(stage %d/%d, landmark %d/%d, tree %d/%d)", mId+1, 7, i+1, mLandmarkNum, j+1, mMaxTreeNum);
			vecSampleIndex.clear();
			indexStart = std::max((int)(j * samplePerTree * (1 - mSampleOverlapRate)), 0);
			indexEnd = std::min(indexStart + samplePerTree, sampleCount);
			for (int k = indexStart; k < indexEnd; k++)
			{
				vecSampleIndex.push_back(k);
			}
			mVecTree[i][j].Train(trainData, vecSampleIndex);
		}
	}
}

void FDRandomForest::Read(std::ifstream& fs)
{
	fs.read((char *)&mId, sizeof(int));
	fs.read((char *)&mLandmarkNum, sizeof(int));
	fs.read((char *)&mMaxTreeNum, sizeof(int));
	fs.read((char *)&mSampleOverlapRate, sizeof(double));

	mVecTree.resize(mLandmarkNum);
	for (int i = 0; i < mLandmarkNum; i++)
	{
		std::vector<FDRandomTree> &curVecTree = mVecTree[i];
		curVecTree.resize(mMaxTreeNum);
		for (int j = 0; j < mMaxTreeNum; j++)
		{
			curVecTree[j].Read(fs);
		}
	}
}

void FDRandomForest::Write(std::ofstream& fs)
{
	fs.write((const char *)&mId, sizeof(int));
	fs.write((const char *)&mLandmarkNum, sizeof(int));
	fs.write((const char *)&mMaxTreeNum, sizeof(int));
	fs.write((const char *)&mSampleOverlapRate, sizeof(double));

	for (int i = 0; i < mLandmarkNum; i++)
	{
		std::vector<FDRandomTree> &curVecTree = mVecTree[i];
		for (int j = 0; j < mMaxTreeNum; j++)
		{
			curVecTree[j].Write(fs);
		}
	}
}
