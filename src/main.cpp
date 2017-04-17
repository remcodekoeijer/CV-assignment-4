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
vector<Rect> get_sliding_windows(Mat& image, Size win, Ptr<SVM> svm, vector<Mat>& outResults);
vector<Rect> non_maximum_suppression(vector<Rect> boundingBoxes, float overlap, vector<Mat> outResults);
float area_overlapping_rects(int xLength, int yLength, Rect r1, Rect r2);

int main()
{
	
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
	//imshow("testResizeNeg", resizedBoundingImages[49]); //weird not shown correclty

	//hog stuff them-------------------------------------------------------------
	Size hogWinStride = Size(16, 16);
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
	int labels[150];
	for (int i = 0; i < 150; i++)
	{
		if (i < numberOfImagesPos)
			labels[i] = 1;
		else
			labels[i] = -1;
	}
	
	

	Mat labelsMat(150, 1, CV_32SC1, labels);
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
	svm->setType(SVM::C_SVC);
	//svm->setCoef0(0.0);
	//svm->setDegree(3);
	//svm->setGamma(0);
	//svm->setNu(0.5);
	//svm->setP(0.01); // for EPSILON_SVR, epsilon in loss function?
	//svm->setC(100.01); // From paper, soft classifier
	svm->setKernel(SVM::CHI2);
	//svm->setType(SVM::EPS_SVR);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	//train the svm
    svm->trainAuto(tData, 10);

	//svm->train(tData);
	

	//save it
	svm->save("svm.xml");

	//predict-----------------------------------------------
	Mat testImageNeg2 = imread("data/pug_188.jpg", IMREAD_GRAYSCALE);
	Mat testResult2;
	testImageNeg2.copyTo(testResult2);
	resize(testResult2, testResult2, Size(128, 128));
	vector<float> descriptorsN2(1);
	vector<Point> locations2(1);
	hog.compute(testResult2, descriptorsN2, hogWinStride, hogPadding, locations2);
	cout << "predict " << svm->predict(descriptorsN2) << endl;
	//----------------------------------------------------------------

	for (int num = 1; num < 101; num++)
	{
		//string name = "data/test22.jpg";

		string name = "data/test" + to_string(num) + ".jpg";

		Mat scaledOrig = imread(name, IMREAD_GRAYSCALE);
		Mat scaledOrig2 = imread(name, IMREAD_GRAYSCALE);
		Mat scaledOrig3 = imread(name, IMREAD_GRAYSCALE);
		Size slidingWindowSize = Size(128, 128);

		vector<Mat> imagePyramid;
		Mat scaled1, scaled2, scaled3, scaled4, scaled5;
		float scaleSize = 1.3f;

		resize(scaledOrig, scaled1, Size(scaledOrig.cols / scaleSize, scaledOrig.rows / scaleSize));
		resize(scaled1, scaled2, Size(scaled1.cols / scaleSize, scaled1.rows / scaleSize));
		resize(scaled2, scaled3, Size(scaled2.cols / scaleSize, scaled2.rows / scaleSize));
		resize(scaled3, scaled4, Size(scaled3.cols / scaleSize, scaled3.rows / scaleSize));
		resize(scaled4, scaled5, Size(scaled4.cols / scaleSize, scaled4.rows / scaleSize));

		imagePyramid.push_back(scaledOrig);
		imagePyramid.push_back(scaled1);
		imagePyramid.push_back(scaled2);
		imagePyramid.push_back(scaled3);
		imagePyramid.push_back(scaled4);
		imagePyramid.push_back(scaled5);

		vector<vector<Rect>> scaledRects;
		vector<Rect> scaledRectsResized;

		vector<Mat> outResults;
		vector<Rect> getWindowsOrig = get_sliding_windows(scaledOrig, slidingWindowSize, svm, outResults);
		vector<Rect> getWindows1 = get_sliding_windows(scaled1, slidingWindowSize, svm, outResults);
		vector<Rect> getWindows2 = get_sliding_windows(scaled2, slidingWindowSize, svm, outResults);
		vector<Rect> getWindows3 = get_sliding_windows(scaled3, slidingWindowSize, svm, outResults);
		vector<Rect> getWindows4 = get_sliding_windows(scaled4, slidingWindowSize, svm, outResults);
		vector<Rect> getWindows5 = get_sliding_windows(scaled5, slidingWindowSize, svm, outResults);


		scaledRects.push_back(getWindowsOrig);
		scaledRects.push_back(getWindows1);
		scaledRects.push_back(getWindows2);
		scaledRects.push_back(getWindows3);
		scaledRects.push_back(getWindows4);
		scaledRects.push_back(getWindows5);

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
					rectangle(imagePyramid[i], *loc, Scalar(255, 255, 255), 2, 8, 0); //image pyramid

					Rect rectResized = Rect(loc->x * scaledFactor, loc->y * scaledFactor, loc->width * scaledFactor, loc->height * scaledFactor);
					scaledRectsResized.push_back(rectResized);
					rectangle(scaledOrig2, rectResized, Scalar(255, 255, 255), 2, 8, 0);
					//cout << "test" << endl;
					count++;
				}
			}
			//imshow("window" + i, imagePyramid[i]);
		}
		//	cout << "out size " << outResults.size() << " rects size " << scaledRectsResized.size() <<" test "<< outResults[0].at<float>(0) <<  " test " << outResults[1].at<float>(0)<< endl;
		//imshow("windowAllRectangle", scaledOrig2);
		imwrite("data/outputAll/windowAllRectangle" + to_string(num) + ".jpg", scaledOrig2);



		vector<Rect> nmsRect;
		if (!scaledRectsResized.empty())
			nmsRect = non_maximum_suppression(scaledRectsResized, 0.5f, outResults);

		cout << "nmsRect size " << nmsRect.size() << endl;

		for (int i = 0; i < nmsRect.size(); i++)
		{
			rectangle(scaledOrig3, nmsRect[i], Scalar(255, 255, 255), 2, 8, 0);
		}

		//imshow("nms", scaledOrig3);
		imwrite("data/outputAll/nms" + to_string(num) + ".jpg", scaledOrig3);

	}
	waitKey(0);
	return 0;
}

vector<Rect> non_maximum_suppression(vector<Rect> boundingBoxes, float overlap, vector<Mat> outResults)
{
	//TODO
	/*
	Get the best rectangle / sort them (what is the best rectangle? largest size?)
	Loop over other rectangles and see how much they overlap (intersection over union)
	If exceeding overlap, throw them away
	Add the rectangle to result and remove from list
	Repeat on remaining rectangles
	*/

	int bestIdx = 0;
	int largestArea = boundingBoxes[0].area();

	for (int i = 1; i < boundingBoxes.size(); i++)
	{
		if (boundingBoxes[i].area() > largestArea)
		{
			bestIdx = i;
			largestArea = boundingBoxes[i].area();
		}
	}

	//set the largest area rect in the first index
	iter_swap(boundingBoxes.begin(), boundingBoxes.begin() + bestIdx);
	
	Rect r = boundingBoxes[0]; //use the boundingbox with the largest area
	
	vector<Rect> result; //vector with the resulting boundingboxes

	vector<Rect> firstResult;
	vector<float> firstOut;


	firstResult.push_back(boundingBoxes[0]);
	firstOut.push_back(outResults[0].at<float>(0));

	cout << "first result " << firstResult[0] << endl;

	boundingBoxes.erase(boundingBoxes.begin());
	outResults.erase(outResults.begin());

	
	for (int i = 0; i < boundingBoxes.size();)
	{
		
		Rect r2 = boundingBoxes[i];
		float intersectionArea = 0;

		//case with 1 rect encapsulated by the other
		if (r.x < r2.x && r.y < r2.y &&
			r2.x + r2.width < r.x + r.width &&
			r2.y + r2.height < r.y + r.height)
		{
			//r2 is completely inside r
			intersectionArea = r2.area();
		}
		else if (r.x > r2.x && r.y > r2.y &&
			r2.x + r2.width > r.x + r.width &&
			r2.y + r2.height > r.y + r.height)
		{
			//r is completely inside r2
			intersectionArea = r.area();
		}
		
		/*
		//case where 2 corners are inside the other rect
		else if (r.x < r2.x && r.y < r2.y)
		{
			if (r2.x + r2.width < r.x + r.width)
			{
				//r2 has 2 corners inside r and sticks out in y direction
				intersectionArea = r2.width * (r.height - std::abs(r.y - r2.y));
			}
			else if (r2.y + r2.height < r.y + r.height)
			{
				//r2 has 2 corners inside r and sticks out in x direction
				intersectionArea = (r.width - std::abs(r.x - r2.x)) * r2.height;
			}
		}

		else if (r.x > r2.x && r.y > r2.y)
		{
			if (r2.x + r2.width > r.x + r.width)
			{
				//r has 2 corners inside r2 and sticks out in y direction
				intersectionArea = r.width * (r2.height - std::abs(r2.y - r.y));
			}
			else if (r2.y + r2.height > r.y + r.height)
			{
				//r has 2 corners inside r2 and sticks out in x direction
				intersectionArea = (r2.width - std::abs(r2.x - r.x)) * r.height;
			}
		}		
		*/

		//case with 1 corner inside the other rect
		else if (r.x < r2.x)
		{
			if (r.y < r2.y)
				intersectionArea = area_overlapping_rects(r.width, r.height, r, r2);
			else
				intersectionArea = area_overlapping_rects(r.width, r2.height, r, r2);
		}
		else
		{
			if (r.y < r2.y)
				intersectionArea = area_overlapping_rects(r2.width, r.height, r, r2);
			else
				intersectionArea = area_overlapping_rects(r2.width, r2.height, r, r2);
		}

		float rectUnion = r.area() + r2.area() - intersectionArea;

		float iou = (intersectionArea / rectUnion);
		
		if (iou > overlap)
		{
			//boundingBoxes.erase(boundingBoxes.begin() + i);

			//add to new vector mat
			firstResult.push_back(boundingBoxes[i]);
			firstOut.push_back(outResults[i].at<float>(0));
			i++;
		
		}
		else
		{
			i++;
		}
		
	}
	cout << "frst out size " << firstOut.size() << endl;
	//---------get maximum
	float min = firstOut[0];//check if maximum
	Rect bestRect= firstResult[0];
	
	for (int j = 0; j < firstOut.size(); j++)
	{
		if (firstOut[j] < min)
		{
			min = firstOut[j];
			bestRect = firstResult[j]; //get it and add it
			cout << "min " << min << endl;
		}

	}
	
	//----------------------
	cout << "size 1 " << boundingBoxes.size() << endl;
	//---------revome from bounding boxes
	for (int j = 0; j < outResults.size(); j++)
	{
		for (int jj = 0; jj < firstOut.size(); jj++)
		{
			if (outResults[j].at<float>(0) == firstOut[jj])
			{
				boundingBoxes.erase(boundingBoxes.begin() + j);
				outResults.erase(outResults.begin() + j);
			}
		}
	}
	cout << "size 2 " << boundingBoxes.size() << endl;
	//----------------------------------
	
	if (boundingBoxes.size() > 1)
	{
		cout << "test" << endl;
		result = non_maximum_suppression(boundingBoxes, overlap, outResults);
		
	}
	result.push_back(bestRect);
	return result;
}

float area_overlapping_rects(int xLength, int yLength, Rect r1, Rect r2)
{
	return (xLength - std::abs(r1.x - r2.x)) * (yLength - std::abs(r1.y - r2.y));
}

vector<Rect> get_sliding_windows(Mat& image, Size win, Ptr<SVM> svm, vector<Mat>& outResults)
{
	vector<Rect> rects;
	int step = 16;
	float threshold = -0.1;
	int winWidth = win.width;
	int winHeight = win.height;
	//create hog for prediction, same parameters as before
	HOGDescriptor hog;
	hog.winSize = Size(128, 128);
	vector<float> descriptorsN2(1);
	vector<Point> locations2(1);
	for (int i = 0; i<image.rows; i += step) //bassically just slides a fixed window
	{
		if ((i + winHeight)>image.rows) { break; }
		for (int j = 0; j< image.cols; j += step)
		{
			if ((j + winWidth)>image.cols) { break; }
			Rect rect(j, i, winWidth, winHeight);
			//hog compute and svm
			Mat subImg = image(rect);
			hog.compute(subImg, descriptorsN2, Size(16, 16), Size(0, 0), locations2);
			Mat out;
			svm->predict(descriptorsN2, out,true); //predict with decision function value
			if (out.at<float>(0) < threshold)
			{
				cout << "predict out" << out << endl;
				outResults.push_back(out);
				rects.push_back(rect); //if above threshold add on all rectangles
			}
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