// FaceAlignmentAPP.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "FDCVInclude.h"
#include "FDUtility.h"
#include "FDLocalBinaryFeatureModel.h"
#include <fstream>
#include <iomanip>

void TrainModel()
{
	FDTrainData trainData;
	FDLocalBinaryFeatureModelParam param;
	FDLocalBinaryFeatureModel model;
	std::string strDir = FD_TEMP_DIR;
	std::string cascadeClassifierModelPath = strDir + "haarcascade_frontalface_alt.xml";
	std::vector<std::string> vecPath;
	//vecPath.push_back(strDir + "list_afw.txt");
	vecPath.push_back(strDir + "list_helen.txt");
	//vecPath.push_back(strDir + "list_lfpw.txt");
	//vecPath.push_back(strDir + "list_ibug.txt");
	//vecPath.push_back(strDir + "list_300windoor.txt");
	//vecPath.push_back(strDir + "list_300woutdoor.txt");

	std::vector<std::string> vecImgPath;
	FDUtility::GenerateTrainData(vecPath, cascadeClassifierModelPath, param.mShapeGenerateNumPerSample, trainData, &vecImgPath);

	model.SetCascadeClassifierModelPath(cascadeClassifierModelPath.c_str());
	model.Train(param, trainData);

	model.Save(FDUtility::StdStringFormat(std::string(), "%s/model/model.dat", FD_TEMP_DIR).c_str());
}

void LoadModel(FDLocalBinaryFeatureModel &model)
{
	std::string strDir = FD_TEMP_DIR;
	std::string cascadeClassifierModelPath = strDir + "haarcascade_frontalface_alt.xml";
	model.SetCascadeClassifierModelPath(cascadeClassifierModelPath.c_str());
	model.Load(FDUtility::StdStringFormat(std::string(), "%s/model/model.dat", FD_TEMP_DIR).c_str());
}

void TestModel()
{
	std::string strDir = FD_TEMP_DIR;
	std::vector<std::string> vecPath;
	//vecPath.push_back(strDir + "list_helen_test.txt");
	//vecPath.push_back(strDir + "list_afw.txt");
	vecPath.push_back(strDir + "list_ibug.txt");
	vecPath.push_back(strDir + "list_lfpw_test.txt");
	//vecPath.push_back(strDir + "list_300windoor.txt");
	//vecPath.push_back(strDir + "list_300woutdoor.txt");

	std::string tempPath;
	std::vector<std::string> vecImgPath;
	vecImgPath.push_back(std::string(FD_TEMP_DIR)+"../TestData/img/1.jpg");
	int listCount = (int)vecPath.size();

	for (int i = 0; i < listCount; i++)
	{
		std::ifstream fin;
		fin.open(vecPath[i]);

		while (std::getline(fin, tempPath))
		{
			vecImgPath.push_back(tempPath);
		}
	}

	FDLocalBinaryFeatureModel model;
	LoadModel(model);

	int testCount = (int)vecImgPath.size();
	for (int i = 0; i < testCount; i++)
	{
		FDLog("test: %s (%d/%d)", vecImgPath[i].c_str(), i+1, testCount);
		cv::Mat_<uchar> test = cv::imread(vecImgPath[i], cv::IMREAD_GRAYSCALE);
		cv::Mat testColor = cv::imread(vecImgPath[i]);
		std::vector<cv::Mat_<double> > result;
		if (model.Predict(test, result))
		{
			int count = (int)result.size();
			for (int j = 0; j < count; j++)
			{
				FDUtility::DrawShape(result[j], testColor, 255);
				cv::imwrite(FDUtility::StdStringFormat(std::string(), "%s/result/%d.jpg", FD_TEMP_DIR, i).c_str(), testColor);
				/*
				FDLog("rows: %d", result.rows);
				for (int j = 0; j < result.rows; j++)
				{
				FDLog("(%d %d) = (%d %d)", i, j, (int)result(j, 0), (int)result(j, 1));
				}
				*/
			}
		}
		else
		{
			FDLog("%s not detect face", vecImgPath[i].c_str());
		}
	}
}

void SaveLandmarkToFile(const char *path, const cv::Mat_<double> &shape)
{
	std::ofstream fs(path);
	fs << "version: 1" << std::endl;
	fs << "n_points:  68" << std::endl;
	fs << "{" << std::endl;
	for (int i = 0; i < shape.rows; i++)
	{
		fs << std::fixed << std::setprecision(6) <<shape(i, 0) << " " << std::fixed << std::setprecision(6) << shape(i, 1) << std::endl;
	}
	fs << "}";
}

void TestWithCamera()
{
	FDLocalBinaryFeatureModel model;
	LoadModel(model);

	std::string strDir = std::string(FD_TEMP_DIR) + "/temp/";
	std::string strOutputName = "photo";
	std::string strPath = strDir + strOutputName + ".png";
	std::string strPathLandmark = strDir + strOutputName +".pts";
	std::string strPathWithLandmark = strDir + strOutputName + "withlandmark.png";

	cv::VideoCapture videoCapture;
	videoCapture.open(0);
	if (!videoCapture.isOpened())
	{
		FDLog("camera open failed!");
		return;
	}
	videoCapture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
	videoCapture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
	cv::Mat frame;
	cv::Mat saveFrame;
	cv::Mat gray;
	while (true)
	{
		videoCapture >> frame;
		cv::flip(frame, frame, 1);
		saveFrame = frame.clone();
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		std::vector<cv::Mat_<double> > result;
		std::vector<FDBoundingBox> vecBox;
		if (model.Predict(gray, result, &vecBox))
		{
			int count = (int)result.size();
			for (int j = 0; j < count; j++)
			{
				FDUtility::DrawShape(result[j], frame, 255);
				cv::rectangle(frame, cv::Rect(vecBox[j].m_x, vecBox[j].m_y, vecBox[j].m_width, vecBox[j].m_height), cv::Scalar(255, 0, 0));
				//cv::rectangle(frame, cv::Rect(vecBox[j].m_x-100, vecBox[j].m_y-100, vecBox[j].m_width+200, vecBox[j].m_height+200), cv::Scalar(0, 255, 0));
			}
		}

		cv::imshow("result", frame);
		char ch = cv::waitKey(5);
		if (ch == 's' && !result.empty())
		{
			cv::imwrite(strPath, saveFrame);
			cv::imwrite(strPathWithLandmark, frame);
			SaveLandmarkToFile(strPathLandmark.c_str(), result[0]);
		}
	}
}

int main()
{
	int type = 2;
	switch (type)
	{
	case 0:
		TrainModel();
		break;
	case 1:
		TestModel();
		break;
	case 2:
		TestWithCamera();
		break;
	default:
		TestWithCamera();
		break;
	}
	FDLog("process end");
	cv::waitKey();
    return 0;
}

