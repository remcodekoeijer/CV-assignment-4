#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <vector>
#include <math.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size);
void getBoundingBox(string name, int& ymin, int& ymax, int& xmin, int& xmax);
void createBoundingBoxImage(Mat img, Mat &bimg, int bbYMin, int bbYMax, int bbXMin, int bbXMax);
void getNameOfImages(vector<string>& nameOfImages, string xmlname, int& num);
vector<Rect> get_sliding_windows(Mat& image, Size win, Ptr<SVM> svm);
vector<Rect> non_maximum_suppression(vector<Rect> boundingBoxes, float overlap);

int main()
{
	//TODO:
	/*
	Load an image from the dataset.
	Read the bounding box from the xml file
	Show the image with bounding box
	Extract hog features
	Show the hog features in a window for verification

	Create good positive and negative training sets
	Set up SVM with the entire training set
	Verify overfitting (cross validation)
	Verify the training model? How?

	Create a testing set
	Search for the window with the highest score in an image and apply non-maximum suppresion
	Done, i guess

	*/
	//get images names--------------------------------------------------------------
	vector<string> nameOfImagesPos;
	vector<string> nameOfImagesNeg;
	int numberOfImagesPos, numberOfImagesNeg;
	getNameOfImages(nameOfImagesPos, "data/namePos.xml", numberOfImagesPos);
	getNameOfImages(nameOfImagesNeg, "data/nameNeg.xml", numberOfImagesNeg);

	//store those images------------------------------------------------------------
	vector<Mat> allImgsPos;
	vector<Mat> allImgsNeg;
	for (int i = 0; i < numberOfImagesPos; i++)
	{
		allImgsPos.push_back(imread("data/positive/" + nameOfImagesPos[i], IMREAD_GRAYSCALE));
	}
	
	for (int i = 0; i < numberOfImagesNeg; i++)
	{
		allImgsNeg.push_back(imread("data/negative/" + nameOfImagesNeg[i], IMREAD_GRAYSCALE));
	}
	
	//create boundingboxed images for positive sample-------------------------------
	vector<Mat> boundingImages;
	for (int i = 0; i < numberOfImagesPos; i++)
	{
		int pos = nameOfImagesPos[i].find(".");
		string sub = nameOfImagesPos[i].substr(0, pos);
		int bbYMin, bbYMax, bbXMin, bbXMax;//create variables for bounding box
		getBoundingBox("data/annotations/"+sub+".xml", bbYMin, bbYMax, bbXMin, bbXMax);//get the variables
		Mat bimg(bbYMax - bbYMin, bbXMax - bbXMin, CV_8UC1);
		createBoundingBoxImage(allImgsPos[i], bimg, bbYMin, bbYMax, bbXMin, bbXMax);
		boundingImages.push_back(bimg);
	}

	//resize the bounding images and the negative sample ones--------------------------------------
	vector<Mat> resizedBoundingImages(numberOfImagesPos);  //use these on hog
	vector<Mat> resizedNegImages(numberOfImagesNeg);
	Size smallSize(128, 128);//set values to resize, whaaat values shoud we have? ....................same as winsize is a must?

	for (int i = 0; i < numberOfImagesPos; i++)
	{
		resize(boundingImages[i], resizedBoundingImages[i], smallSize);
	}
	for (int i = 0; i < numberOfImagesNeg; i++)
	{
		resize(allImgsNeg[i], resizedNegImages[i], smallSize);
	}
	//imshow("testResizeNeg", resizedNegImages[0]); //weird not shown correclty

	//hog stuff them-------------------------------------------------------------
	Size hogWinStride = Size(8, 8);
	Size hogPadding = Size(0, 0);

	vector<vector<float>> descriptorsP(numberOfImagesPos);
	vector<vector<float>> descriptorsN(numberOfImagesNeg);
	vector<vector<Point>> locationsPos(numberOfImagesPos); 
	vector<vector<Point>> locationsNeg(numberOfImagesNeg); 
	HOGDescriptor hog;

	hog.winSize = Size(128, 128); //size of the window to get the hog? so bounding box or entire image?

	for (int i = 0; i < numberOfImagesPos; i++)
	{
		hog.compute(resizedBoundingImages[i], descriptorsP[i], hogWinStride, hogPadding, locationsPos[i]);
	}

	for (int i = 0; i < numberOfImagesNeg; i++)
	{
		hog.compute(resizedNegImages[i], descriptorsN[i], hogWinStride, hogPadding, locationsNeg[i]);
	}
	
	//create labels
	//hardcoded for now will change
	int labels[85];
	for (int i = 0; i < 85; i++)
	{
		if (i < numberOfImagesPos)
			labels[i] = 1;
		else
			labels[i] = -1;
	}
	
	//vector<int> labels;
	//for (int i = 0; i < numberOfImagesPos + numberOfImagesNeg; i++)
	//{
	//	if (i < numberOfImagesPos)
	//		labels.push_back(1);
	//	else
	//		labels.push_back(-1);
	//}

	Mat labelsMat(85, 1, CV_32SC1, labels);
	//pass them on the training method
	Mat descriptorsT(numberOfImagesPos + numberOfImagesNeg, descriptorsP[0].size(), CV_32FC1);
	//pos
	for (int i = 0; i < numberOfImagesPos ; i++)
	{
		Mat test(descriptorsP[i], true);
		transpose(test, test);
		test.row(0).copyTo(descriptorsT.row(i));
	}
	//neg
	for (int i = 0; i < numberOfImagesNeg; i++)
	{
		Mat test(descriptorsN[i], true);
		transpose(test, test);
		test.row(0).copyTo(descriptorsT.row(i+ numberOfImagesPos));
	}
	
	Ptr<TrainData> tData = TrainData::create(descriptorsT, ROW_SAMPLE, labelsMat);//create it or pass directly to the training part ,train<SVM>(trainingDataMat, ROW_SAMPLE, labelsMat, params);
	
	Ptr<SVM> svm = SVM::create();
	///set parameters of svn
	//svm->setType(SVM::C_SVC);
	//svm->setCoef0(0.0);
	//svm->setDegree(3);
	//svm->setGamma(0);
	//svm->setNu(0.5);
	//svm->setP(0.01); // for EPSILON_SVR, epsilon in loss function?
	svm->setC(0.1); // From paper, soft classifier
	svm->setKernel(SVM::LINEAR);
	//svm->setType(SVM::EPS_SVR);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	//train the svm
	svm->trainAuto(tData, 10);

	//svm->train(tData);
	
	////get support vectors for the hogdescriptor
	//Mat sv = svm->getSupportVectors();
	//vector<float> hog_detector;
	//const int sv_total = sv.rows;
	//// get the decision function
	//Mat alpha, svidx;
	//double rho = svm->getDecisionFunction(0, alpha, svidx);
	//CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	//CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) || (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	//CV_Assert(sv.type() == CV_32F);
	//hog_detector.clear();
	//hog_detector.resize(sv.cols + 1);
	//memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
	//hog_detector[sv.cols] = (float)-rho;
	//hog.setSVMDetector(hog_detector);
	////detection
	//vector<Rect> foundLocations;
	//vector<double> foundWeights;
	//Mat testImagePos = imread("data/positive/pug_101.jpg", IMREAD_GRAYSCALE);
	//Mat testImageNeg = imread("data/negative/neg16.jpg", IMREAD_GRAYSCALE);
	//
	//Mat testResult;
	//testImagePos.copyTo(testResult);
	//// explanation of detectmultiscale parameters 
	//// http://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/
	//// strangely enough, a smaller winstride doesn't give us more detections. 
	//hog.detectMultiScale(testResult, foundLocations, foundWeights, 0, Size(16, 16), Size(0, 0), 1.05, 2.0, false);
	//cout << foundLocations.size() << endl;
	//double maxWeight = 0;
	//int idx = -1;
	////get the index for the highest weight
	//for (int i = 0; i < foundWeights.size(); i++)
	//{
	//	if (foundWeights[i] > maxWeight)
	//	{
	//		maxWeight = foundWeights[i];
	//		idx = i;
	//	}
	//}
	////draw all rectangles
	//*for (int i = 0; i < foundLocations.size(); i++)
	//{
	//	rectangle(testResult, foundLocations[i], Scalar(1, 1, 1, 1), 1, 8, 0);
	//}*/
	////draw the rectangle with the highest weight
	//rectangle(testResult, foundLocations[idx], Scalar(1, 1, 1, 1), 1, 8, 0);
	//imshow("testresult", testResult);


	//predict?------------------------------------------------
	Mat testImageNeg2 = imread("data/pug_188.jpg", IMREAD_GRAYSCALE);
	Mat testResult2;
	testImageNeg2.copyTo(testResult2);
	resize(testResult2, testResult2, Size(128, 128));
	vector<float> descriptorsN2(1);
	vector<Point> locations2(1);
	hog.compute(testResult2, descriptorsN2, hogWinStride, hogPadding, locations2);
	cout << "predict " << svm->predict(descriptorsN2) << endl;
	//----------------------------------------------------------------



	Mat scaledOrig = imread("data/pug_10.jpg", IMREAD_GRAYSCALE);
	Mat scaledOrig2 = imread("data/pug_10.jpg", IMREAD_GRAYSCALE);
	Size slidingWindowSize = Size(128, 128);

	vector<Mat> imagePyramid;
	Mat scaled1, scaled2, scaled3;
	float scaleSize = 1.5f;

	resize(scaledOrig, scaled1, Size(scaledOrig.cols / scaleSize, scaledOrig.rows / scaleSize));
	resize(scaled1, scaled2, Size(scaled1.cols / scaleSize, scaled1.rows / scaleSize));
	resize(scaled2, scaled3, Size(scaled2.cols / scaleSize, scaled2.rows / scaleSize));

	imagePyramid.push_back(scaledOrig);
	imagePyramid.push_back(scaled1);
	imagePyramid.push_back(scaled2);
	imagePyramid.push_back(scaled3);

	vector<vector<Rect>> scaledRects;
	vector<Rect> scaledRectsResized;

	vector<Rect> getWindowsOrig = get_sliding_windows(scaledOrig, slidingWindowSize, svm);
	vector<Rect> getWindows1 = get_sliding_windows(scaled1, slidingWindowSize, svm);
	vector<Rect> getWindows2 = get_sliding_windows(scaled2, slidingWindowSize, svm);
	vector<Rect> getWindows3 = get_sliding_windows(scaled3, slidingWindowSize, svm);

	scaledRects.push_back(getWindowsOrig);
	scaledRects.push_back(getWindows1);
	scaledRects.push_back(getWindows2);
	scaledRects.push_back(getWindows3);

	for (int i = 0; i < scaledRects.size(); i++)
	{
		float scaledFactor = powf(scaleSize, i);

		if (!scaledRects[i].empty())
		{
			vector< Rect >::const_iterator loc = scaledRects[i].begin();
			vector< Rect >::const_iterator end = scaledRects[i].end();
			int count = 0;
			for (; loc != end; ++loc)
			{
				//cout << "weights " << getWindows[count] << endl;
				//if(foundWeights[count]<1)
				rectangle(imagePyramid[i], *loc, Scalar(255, 255, 255), 5, 8, 0); //image pyramid

				Rect rectResized = Rect(loc->x * scaledFactor, loc->y * scaledFactor, loc->width * scaledFactor, loc->height * scaledFactor);
				scaledRectsResized.push_back(rectResized);
				rectangle(scaledOrig2, rectResized, Scalar(255, 255, 255), 5, 8, 0);
				//cout << "test" << endl;
				count++;
			}
		}
		imshow("window" + i, imagePyramid[i]);
	}
	imshow("windowAllRectangle", scaledOrig2);

	vector<Rect> nmsRect = non_maximum_suppression(scaledRectsResized, 0.5f);
	for (int i = 0; i < nmsRect.size(); i++)
	{
		rectangle(scaledOrig2, nmsRect[i], Scalar(255, 255, 255), 5, 8, 0);
	}

	//save it
	svm->save("svm.xml");

	waitKey(0);
	return 0;
}

vector<Rect> non_maximum_suppression(vector<Rect> boundingBoxes, float overlap)
{
	//TODO
	/*
	Get the best rectangle / sort them (what is the best rectangle? largest size?)
	Loop over other rectangles and see how much they overlap (intersection over union)
	If exceeding overlap, throw them away
	Add the rectangle to result and remove from list
	Repeat on remaining rectangles
	*/




	vector<Rect> result;

	return result;
}

vector<Rect> get_sliding_windows(Mat& image, Size win, Ptr<SVM> svm)
{
	vector<Rect> rects;
	int step = 16;
	int winWidth = win.width;
	int winHeight = win.height;
	for (int i = 0; i<image.rows; i += step)
	{
		if ((i + winHeight)>image.rows) { break; }
		for (int j = 0; j< image.cols; j += step)
		{
			if ((j + winWidth)>image.cols) { break; }
			Rect rect(j, i, winWidth, winHeight);
			//hog compute and svm
			Mat subImg = image(rect);
			HOGDescriptor hog;
			hog.winSize = Size(128, 128);
			vector<float> descriptorsN2(1);
			vector<Point> locations2(1);
			hog.compute(subImg, descriptorsN2, Size(8, 8), Size(0, 0), locations2);
			//cout << "predict " << svm->predict(descriptorsN2) << endl;
			Mat out;
			svm->predict(descriptorsN2, out);
			cout << "predict out" << out << endl;
			if (svm->predict(descriptorsN2) == 1)
				rects.push_back(rect);
		}
	}
	return rects;
}

void getNameOfImages(vector<string>& nameOfImages, string xmlname, int& num)//get name of images
{
	FileStorage fs(xmlname, FileStorage::READ);
	fs["num"] >> num; //read number, change this in xml
	for (int i = 1; i < num + 1; i++)
	{
		nameOfImages.push_back(fs["i" + to_string(i)]); //push them back to vector
		//cout << nameOfImages[i - 1] << endl;
	}
	fs.release();
}

void createBoundingBoxImage(Mat img, Mat &bimg, int bbYMin, int bbYMax, int bbXMin, int bbXMax)
{
	int ii = 0; //variables to increment bb image values
	int jj = 0;
	for (int i = bbYMin; i < bbYMax; i++)
	{
		for (int j = bbXMin; j < bbXMax; j++)
		{
			uchar pixel = img.at<uchar>(i, j);//get values from original image
			bimg.at<uchar>(ii, jj) = pixel;
			jj++;
		}
		ii++;
		jj = 0;
	}

}
void getBoundingBox(string name, int& ymin, int& ymax, int& xmin, int& xmax)//set the variables, important xml must have <opencv_storage> on top, check pug_166.xml
{
	FileStorage fs2(name, FileStorage::READ);
	FileNode n = fs2["annotation"];
	FileNode n2 = n["object"];
	FileNode n3 = n2["bndbox"];
	n3["ymin"] >> ymin;
	n3["ymax"] >> ymax;
	n3["xmin"] >> xmin;
	n3["xmax"] >> xmax;
	//cout << "xs an ys are " << xmin << " " << ymin << " " << xmax << " " << ymax << endl;

	fs2.release();
}

//From the train_hog.cpp, for visualizing a hogdescriptor. 
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size)
{
	const int DIMX = size.width;
	const int DIMY = size.height;
	float zoomFac = 3;
	Mat visu;
	resize(color_origImg, visu, Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac)));

	int cellSize = 8;
	int gradientBinSize = 9;
	float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?

																	   // prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				cellx = blockx;
				celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				  // note: overlapping blocks lead to multiple updates of this sum!
				  // we therefore keep track how often a cell was updated,
				  // to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	  // compute average gradient strengths
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			int mx = drawX + cellSize / 2;
			int my = drawY + cellSize / 2;

			rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), Scalar(100, 100, 100), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)(cellSize / 2.f);
				float scale = 2.5; // just a visualization scale, to see the lines better

								   // compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visualization
				line(visu, Point((int)(x1*zoomFac), (int)(y1*zoomFac)), Point((int)(x2*zoomFac), (int)(y2*zoomFac)), Scalar(0, 255, 0), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	  // don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;

} // get_hogdescriptor_visu