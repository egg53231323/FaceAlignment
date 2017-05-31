#include "stdafx.h"
#include "FDLocalBinaryFeatureModelImp.h"
#include "FDLocalBinaryFeatureModel.h"
#include "FDUtility.h"
#include "liblinear/linear.h"

FDLocalBinaryFeatureModelImp::~FDLocalBinaryFeatureModelImp()
{
	ReleaseModel();
}

void FDLocalBinaryFeatureModelImp::SetCascadeClassifierModelPath(const char *path)
{
	mCascadeClassifier.load(path);
}

void FDLocalBinaryFeatureModelImp::Train(const FDLocalBinaryFeatureModelParam &param, FDTrainData &trainData)
{
	ReleaseModel();
	mVecModels.resize(param.mStageNum);
	int leafFeatureNum = param.mLandmarkNum * param.mMaxTreeNum * (int)pow(2, (param.mMaxTreeDepth - 1));
	mVecRandomForest.clear();
	mVecRandomForest.resize(param.mStageNum);
	int sampleCount = (int)trainData.mVecDataItems.size();
	for (int i = 0; i < param.mStageNum; i++)
	{
		FDRandomForest &currentForest = mVecRandomForest[i];
		currentForest.mId = i;
		GetShapeResidual(trainData);

		currentForest.SetParam(param.mMaxTreeNum, param.mMaxTreeDepth, 
			param.mFeatureGenerateCount[i], param.mFeatureGenerateRadius[i], 
			param.mSampleOverlapRate, param.mLandmarkNum);
		currentForest.Train(trainData);

		feature_node **feature = GenerateFeature(currentForest, trainData);
		GlobalRegression(feature, trainData, mVecModels[i], leafFeatureNum);
		ReleaseFeature(feature, sampleCount);
	}

	mPredictData.mMeanShape = trainData.mMeanShape.clone();
}

bool FDLocalBinaryFeatureModelImp::Predict(const cv::Mat_<uchar> &image, std::vector<cv::Mat_<double> > &result)
{
	uint64 t1 = FDUtility::GetCurrentTime();
	// todo multi
	double scale = 1.3;
	std::vector<cv::Rect> faces;
	cv::Mat smallImg(cvRound(image.rows / scale), cvRound(image.cols / scale), CV_8UC1);
	cv::resize(image, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
	cv::equalizeHist(smallImg, smallImg);

	mCascadeClassifier.detectMultiScale(smallImg, faces, 1.1, 2,
		0
		//|CV_HAAR_FIND_BIGGEST_OBJECT
		//|CV_HAAR_DO_ROUGH_SEARCH
		| CV_HAAR_SCALE_IMAGE
		,
		cv::Size(30, 30));

	uint64 t2 = FDUtility::GetCurrentTime();
	FDLog("cost detect %d", (int)(t2 - t1));
	if (faces.empty())
		return false;

	FDBoundingBox boundingBox;

	int faceCount = (int)faces.size();
	result.resize(faceCount);
	for (int i = 0; i < faceCount; i++)
	{
		boundingBox.m_x = faces[i].x*scale;
		boundingBox.m_y = faces[i].y*scale;
		boundingBox.m_width = (faces[i].width - 1) * scale;
		boundingBox.m_height = (faces[i].height - 1) * scale;
		boundingBox.CalcCenter();

		Predict(image, result[i], boundingBox);
	}
	return true;
}

bool FDLocalBinaryFeatureModelImp::Predict(const cv::Mat_<uchar> &image, cv::Mat_<double> &result, const FDBoundingBox &boudingBox)
{
	uint64 t3 = FDUtility::GetCurrentTime();

	int stageNum = (int)mVecModels.size();
	mPredictData.mVecDataItems.clear();
	mPredictData.mVecDataItems.push_back(FDTrainDataItem());
	FDTrainDataItem &item = mPredictData.mVecDataItems[0];
	item.mBoundingBox = boudingBox;
	item.mImage = image;
	item.mCurrentShape = FDUtility::RelativeToReal(mPredictData.mMeanShape, item.mBoundingBox);
	for (int i = 0; i < stageNum; i++)
	{
		// todo memory 
		feature_node **feature = GenerateFeature(mVecRandomForest[i], mPredictData);
		GlobalPrediction(feature, mVecModels[i], item.mCurrentShape, item.mBoundingBox);
		ReleaseFeature(feature, 1);
	}
	result = item.mCurrentShape;
	mPredictData.mVecDataItems.clear();

	uint64 t4 = FDUtility::GetCurrentTime();
	FDLog("cost predict %d", (int)(t4 - t3));
	return true;
}

void FDLocalBinaryFeatureModelImp::GetShapeResidual(FDTrainData &trainData)
{
	cv::Mat_<double> rotation;
	double scale = 0;
	std::vector<FDTrainDataItem> &vecItem = trainData.mVecDataItems;
	cv::Mat_<double> &meanShape = trainData.mMeanShape;
	int sampleCount = (int)vecItem.size();
	for (int i = 0; i < sampleCount; i++)
	{
		FDTrainDataItem &item = vecItem[i];
		item.mShapeResidual = FDUtility::RealToRelative(item.mGroundTruthShape, item.mBoundingBox) 
			- FDUtility::RealToRelative(item.mCurrentShape, item.mBoundingBox);
		FDUtility::SimilarityTransform(meanShape, FDUtility::RealToRelative(item.mCurrentShape, item.mBoundingBox), rotation, scale);
		cv::transpose(rotation, rotation);
		item.mShapeResidual = scale * item.mShapeResidual * rotation;
	}
}

feature_node** FDLocalBinaryFeatureModelImp::GenerateFeature(const FDRandomForest &randomForest, FDTrainData &trainData)
{
	std::vector<FDTrainDataItem> &vecItem = trainData.mVecDataItems;
	int sampleCount = (int)vecItem.size();
	feature_node **feature = new feature_node*[sampleCount];
	int featureSizePerSample = randomForest.mLandmarkNum * randomForest.mMaxTreeNum + 1;
	for (int i = 0; i < sampleCount; i++)
	{
		feature[i] = new feature_node[featureSizePerSample];
	}

	cv::Mat_<double> rotation;
	double scale = 0;

	cv::Mat_<double> &meanShape = trainData.mMeanShape;
	for (int i = 0; i < sampleCount; i++)
	{
		FDTrainDataItem &item = vecItem[i];
		FDUtility::SimilarityTransform(FDUtility::RealToRelative(item.mCurrentShape, item.mBoundingBox), meanShape, rotation, scale);
		GetCodeFromRandomForest(feature[i], randomForest, item, rotation, scale);
		feature[i][featureSizePerSample - 1].index = -1;
		feature[i][featureSizePerSample - 1].value = -1;
	}

	return feature;
}

void FDLocalBinaryFeatureModelImp::GetCodeFromRandomForest(feature_node *feature, const FDRandomForest &randomForest,
	const FDTrainDataItem &item, const cv::Mat_<double> &rotation, double scale)
{
	const FDBoundingBox &boundingBox = item.mBoundingBox;
	const cv::Mat_<uchar> image = item.mImage;
	int leafNodePerTree = (int)pow(2, randomForest.mVecTree[0][0].mMaxDepth - 1);
	int maxTreeNum = randomForest.mMaxTreeNum;
	int featurePos = 0;
	for (int i = 0; i < randomForest.mLandmarkNum; i++)
	{
		const std::vector<FDRandomTree> &vecCurLandmardTree = randomForest.mVecTree[i];
		int landmarkX = (int)item.mCurrentShape(i, 0);
		int landmarkY = (int)item.mCurrentShape(i, 1);

		for (int j = 0; j < maxTreeNum; j++)
		{
			int curNodeId = 0;
			int code = 1;
			const FDRandomTree &tree = vecCurLandmardTree[j];
			const std::vector<FDNode> &vecNode = tree.mVecNodes;
			// depth is start from 1
			for (int k = 0; k < tree.mMaxDepth - 1; k++)
			{
				const FDNode &curNode = vecNode[curNodeId];
				double x1 = curNode.mFeature[0];
				double y1 = curNode.mFeature[1];
				double x2 = curNode.mFeature[2];
				double y2 = curNode.mFeature[3];

				double tempX1 = rotation(0, 0) * x1 + rotation(0, 1) * y1;
				double tempY1 = rotation(1, 0) * x1 + rotation(1, 1) * y1;
				tempX1 = scale * tempX1 * boundingBox.m_width / 2.0;
				tempY1 = scale * tempY1 * boundingBox.m_height / 2.0;
				int realX1 = (int)tempX1 + landmarkX;
				int realY1 = (int)tempY1 + landmarkY;
				realX1 = std::max(0, std::min(realX1, image.cols - 1));
				realY1 = std::max(0, std::min(realY1, image.rows - 1));

				double tempX2 = rotation(0, 0) * x2 + rotation(0, 1) * y2;
				double tempY2 = rotation(1, 0) * x2 + rotation(1, 1) * y2;
				tempX2 = scale * tempX2 * boundingBox.m_width / 2.0;
				tempY2 = scale * tempY2 * boundingBox.m_height / 2.0;
				int realX2 = (int)tempX2 + landmarkX;
				int realY2 = (int)tempY2 + landmarkY;
				realX2 = std::max(0, std::min(realX2, image.cols - 1));
				realY2 = std::max(0, std::min(realY2, image.rows - 1));

				int diff = (int)(image(realY1, realX1)) - (int)(image(realY2, realX2));
				if (diff < curNode.mThreshold)
				{
					curNodeId = curNode.mChildrenNodesId[0];
				}
				else
				{
					curNodeId = curNode.mChildrenNodesId[1];
					int move = tree.mMaxDepth - 2 - k;
					code += (1 << move);
				}
			}
		
			feature[featurePos].index = leafNodePerTree*featurePos + code;
			feature[featurePos].value = 1;
			featurePos++;
		}
	}
}

void FDLocalBinaryFeatureModelImp::GlobalRegression(feature_node **feature, FDTrainData &trainData, std::vector<model*> &vecModel , int leafFeatureNum)
{
	std::vector<FDTrainDataItem> &vecItem = trainData.mVecDataItems;
	int sampleCount = (int)vecItem.size();
	// shapes_residual: n*(l*2)
	// construct the problem(expect y)
	problem prob;
	prob.l = sampleCount;
	prob.n = leafFeatureNum;
	prob.x = feature;
	prob.bias = -1;

	// construct the parameter
	parameter param;
	param.solver_type = L2R_L2LOSS_SVR_DUAL;
	//  param-> solver_type = L2R_L2LOSS_SVR;
	param.C = 1.0 / sampleCount;
	param.p = 0;
	param.eps = 0.0001;
	//param->eps = 0.001;

	// initialize the y
	int numResidual = vecItem[0].mShapeResidual.rows * 2;
	double **yy = new double*[numResidual];

	for (int i = 0; i<numResidual; i++)
	{
		yy[i] = new double[sampleCount];
	}
	int th = numResidual / 2;
	for (int i = 0; i < sampleCount; i++)
	{
		FDTrainDataItem &item = vecItem[i];
		for (int j = 0; j < numResidual; j++)
		{
			if (j < th)
			{
				yy[j][i] = item.mShapeResidual(j, 0);
			}
			else
			{
				yy[j][i] = item.mShapeResidual(j - th, 1);
			}
		}
	}

	//train
	vecModel.clear();
	vecModel.resize(numResidual);

	for (int i = 0; i < numResidual; i++)
	{
		prob.y = yy[i];
		check_parameter(&prob, &param);
		model* lbfmodel = train(&prob, &param);
		vecModel[i] = lbfmodel;
	}

	// update the current shape and shapes_residual
	double tmp;
	double scale;
	const cv::Mat_<double> &meanShape = trainData.mMeanShape;
	cv::Mat_<double> rotation;
	cv::Mat_<double> deltashape_bar(numResidual / 2, 2);
	cv::Mat_<double> deltashape_bar1(numResidual / 2, 2);
	for (int i = 0; i < sampleCount; i++)
	{
		FDTrainDataItem &item = vecItem[i];
		for (int j = 0; j<numResidual; j++)
		{
			tmp = predict(vecModel[j], feature[i]);
			if (j < th)
			{
				deltashape_bar(j, 0) = tmp;
			}
			else
			{
				deltashape_bar(j - th, 1) = tmp;
			}
		}
		// transfer or not to be decided
		// now transfer
		FDUtility::SimilarityTransform(FDUtility::RealToRelative(item.mCurrentShape, item.mBoundingBox), meanShape, rotation, scale);
		cv::transpose(rotation, rotation);
		deltashape_bar1 = scale * deltashape_bar * rotation;
		item.mCurrentShape = FDUtility::RelativeToReal((FDUtility::RealToRelative(item.mCurrentShape, item.mBoundingBox) + deltashape_bar1), item.mBoundingBox);
	}
}

void FDLocalBinaryFeatureModelImp::GlobalPrediction(feature_node **feature, const std::vector<model*> &vecModel, cv::Mat_<double> &currentShape, const FDBoundingBox &boundingBox)
{
	int numResidual = currentShape.rows * 2;
	int th = numResidual / 2;
	double tmp;
	double scale;
	cv::Mat_<double> rotation;
	cv::Mat_<double> deltashape_bar(numResidual / 2, 2);
	currentShape = FDUtility::RealToRelative(currentShape, boundingBox);
	for (int j = 0; j<numResidual; j++)
	{
		tmp = predict(vecModel[j], feature[0]);
		if (j < th)
		{
			deltashape_bar(j, 0) = tmp;
		}
		else
		{
			deltashape_bar(j - th, 1) = tmp;
		}
	}
	// transfer or not to be decided
	// now transfer
	FDUtility::SimilarityTransform(currentShape, mPredictData.mMeanShape, rotation, scale);
	cv::transpose(rotation, rotation);
	deltashape_bar = scale * deltashape_bar * rotation;
	currentShape = FDUtility::RelativeToReal((currentShape + deltashape_bar), boundingBox);
}

void FDLocalBinaryFeatureModelImp::ReleaseFeature(feature_node **feature, int num)
{
	for (int i = 0; i < num; i++)
	{
		delete[] (feature[i]);
		feature[i] = NULL;
	}
	delete[] feature;
}

void FDLocalBinaryFeatureModelImp::ReleaseModel()
{
	int count1 = (int)mVecModels.size();
	for (int i = 0; i < count1; i++)
	{
		std::vector<model*> &vecModel = mVecModels[i];
		int count2 = (int)vecModel.size();
		for (int j = 0; j < count2; j++)
		{
			free(vecModel[j]->w);
			vecModel[j]->w = NULL;
			free(vecModel[j]->label);
			vecModel[j]->label = NULL;
			free(vecModel[j]);
			vecModel[j] = NULL;
		}
	}
	mVecModels.clear();
}

bool FDLocalBinaryFeatureModelImp::Save(const char *path)
{
	if (mVecRandomForest.empty() || mVecModels.empty())
		return false;

	std::ofstream fs(path, std::ios::binary);
	WriteRandomForest(fs);
	WriteRegression(fs);
	WriteMeanShape(fs);
	fs.close();
	return true;
}

bool FDLocalBinaryFeatureModelImp::Load(const char *path)
{
	std::ifstream fs(path, std::ios::binary);
	ReadRandomForest(fs);
	ReadRegression(fs);
	ReadMeanShape(fs);
	fs.close();
	return (!mVecRandomForest.empty()) && (!mVecModels.empty());
}

void FDLocalBinaryFeatureModelImp::ReadRandomForest(std::ifstream& fs)
{
	mVecRandomForest.clear();
	int forestNum = 0;
	fs.read((char *)&forestNum, sizeof(int));
	mVecRandomForest.resize(forestNum);
	for (int i = 0; i < forestNum; i++)
	{
		mVecRandomForest[i].Read(fs);
	}
}

void FDLocalBinaryFeatureModelImp::WriteRandomForest(std::ofstream& fs)
{
	int forestNum = (int)mVecRandomForest.size();
	fs.write((const char *)&forestNum, sizeof(int));
	for (int i = 0; i < forestNum; i++)
	{
		mVecRandomForest[i].Write(fs);
	}
}

void FDLocalBinaryFeatureModelImp::ReadRegression(std::ifstream& fs)
{
	ReleaseModel();
	int stageNum = 0;
	fs.read((char *)&stageNum, sizeof(int));
	mVecModels.resize(stageNum);
	for (int i = 0; i < stageNum; i++)
	{
		std::vector<model *> &curVecModel = mVecModels[i];
		int modelNum = 0;
		fs.read((char *)&modelNum, sizeof(int));
		curVecModel.resize(modelNum);
		for (int j = 0; j < modelNum; j++)
		{
			curVecModel[j] = load_model_bin(fs);
		}
	}
}

void FDLocalBinaryFeatureModelImp::WriteRegression(std::ofstream& fs)
{
	int stageNum = (int)mVecModels.size();
	fs.write((const char *)&stageNum, sizeof(int));
	for (int i = 0; i < stageNum; i++)
	{
		std::vector<model *> &curVecModel = mVecModels[i];
		int modelNum = (int)curVecModel.size();
		fs.write((const char *)&modelNum, sizeof(int));
		for (int j = 0; j < modelNum; j++)
		{
			save_model_bin(fs, curVecModel[j]);
		}
	}
}

void FDLocalBinaryFeatureModelImp::ReadMeanShape(std::ifstream& fs)
{
	int rows = 0;
	fs.read((char *)&rows, sizeof(int));
	double *val = new double[rows * 2];
	fs.read((char *)val, sizeof(double) * rows * 2);
	mPredictData.mMeanShape = cv::Mat_<double>(rows, 2);
	for (int i = 0; i < rows; i++)
	{
		mPredictData.mMeanShape(i, 0) = val[i * 2];
		mPredictData.mMeanShape(i, 1) = val[i * 2 + 1];
	}
	delete[]val;
	val = NULL;
}

void FDLocalBinaryFeatureModelImp::WriteMeanShape(std::ofstream& fs)
{
	int rows = mPredictData.mMeanShape.rows;
	fs.write((const char *)&rows, sizeof(int));
	double *val = new double[rows * 2];
	for (int i = 0; i < rows; i++)
	{
		val[i*2] = mPredictData.mMeanShape(i, 0);
		val[i*2+1] = mPredictData.mMeanShape(i, 1);
	}
	fs.write((const char *)val, sizeof(double) * rows * 2);
	delete[]val;
	val = NULL;
}
