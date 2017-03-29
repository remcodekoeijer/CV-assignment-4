#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;
using namespace cv::ml;

Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size);
void getBoundingBox(string name, int& ymin, int& ymax, int& xmin, int& xmax);
void createBoundingBoxImage(Mat img, Mat &bimg, int bbYMin, int bbYMax, int bbXMin, int bbXMax);
void getNameOfImages(vector<string>& nameOfImages, string xmlname, int& num);

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
	//get images names
	vector<string> nameOfImagesPos;
	vector<string> nameOfImagesNeg;
	int numberOfImagesPos, numberOfImagesNeg;
	getNameOfImages(nameOfImagesPos, "data/namePos.xml", numberOfImagesPos);
	getNameOfImages(nameOfImagesNeg, "data/nameNeg.xml", numberOfImagesNeg);
	//store those images
	vector<Mat> allImgsPos;
	for (int i = 0; i < numberOfImagesPos; i++)
		allImgsPos.push_back(imread(nameOfImagesPos[i], IMREAD_GRAYSCALE));
	vector<Mat> allImgsNeg;
	for (int i = 0; i < numberOfImagesPos; i++)
		allImgsNeg.push_back(imread(nameOfImagesNeg[i], IMREAD_GRAYSCALE));

	//create boundingboxed images for positive sample
	//resize them and the negative sample ones
	//hog.compute them
	//create labels?
	//pass them on the training method

	Mat img = imread("data/pug_166.jpg", IMREAD_GRAYSCALE);
	Mat img2 = imread("data/Bengal_109.jpg", IMREAD_GRAYSCALE); //snd image
	cout << "width and height " << img.rows << img.cols << endl;
	namedWindow("OpenCV");
	imshow("OpenCV", img);

	int bbYMin, bbYMax, bbXMin, bbXMax;//create variables for bounding box
	getBoundingBox("data/annotations/pug_166.xml", bbYMin, bbYMax, bbXMin, bbXMax);//get the variables
	int bbYMin2, bbYMax2, bbXMin2, bbXMax2;//create variables for bounding box
	getBoundingBox("data/annotations/Bengal_109.xml", bbYMin2, bbYMax2, bbXMin2, bbXMax2);//get the variables
																						  //------------------show bounding box
																						  //for (int i = bbYMin; i <= bbYMax; i++)
																						  //{
																						  //	for (int j = bbXMin; j <= bbXMax; j++)
																						  //	{
																						  //		 img.at<uchar>(i, j)=0;//set bounding box pixels to black
																						  //		
																						  //	}
																						  //	
																						  //}
																						  //imshow("OpenCV2", img);
																						  //-----------------------------------------------
	Mat bimg(bbYMax - bbYMin, bbXMax - bbXMin, CV_8UC1); //create new image with the bounding box, size should be equal to bounding box
	cout << "test" << endl;
	createBoundingBoxImage(img, bimg, bbYMin, bbYMax, bbXMin, bbXMax);
	cout << "test" << endl;
	Mat bimg2(bbYMax2 - bbYMin2, bbXMax2 - bbXMin2, CV_8UC1); //create new image with the bounding box, size should be equal to bounding box
	createBoundingBoxImage(img2, bimg2, bbYMin2, bbYMax2, bbXMin2, bbXMax2);
	cout << "test" << endl;
	//show
	imshow("bbImage", bimg);
	//resize Image
	Mat rbimg;
	Mat rbimg2;
	Size resized(300, 350);//set values to resize
	resize(bimg, rbimg, resized);
	resize(bimg2, rbimg2, resized);
	imshow("bbImageResized", rbimg);
	imshow("bbImageResized2", rbimg2);

	//------------------------------hog stuff-----------------------
	Size hogWinStride = Size(8, 8);
	Size hogPadding = Size(0, 0);

	vector<vector<float>> descriptors(2);//make it vector<vector<float>>? for multiple images
	vector<Point> locations;
	vector<Point> locations2;

	HOGDescriptor hog;
	hog.winSize = Size(128, 128); //size of the window to get the hog? so bounding box or entire image?

								  //locations is empty during training, because you don't find the location of the object you want to detect, you only extract features.
	hog.compute(rbimg, descriptors[0], hogWinStride, hogPadding, locations); ///leftttttttttt
	hog.compute(rbimg2, descriptors[1], hogWinStride, hogPadding, locations2); ///leftttttttttt
																			   //the descriptors here should be sufficient for training. 
																			   //useful link https://github.com/DaHoC/trainHOG and also check the train_HOG.cpp in the opencv examples.

	cout << hog.winSize.width << endl;
	cout << hog.winSize.height << endl;
	cout << "descriptors list size: " << descriptors.size() << endl;

	//imshow("gradient", get_hogdescriptor_visu(rbimg.clone(), descriptors[0], Size(128, 128)));

	//for (int i=0;i<descriptors.size();i++)
	//	cout << "descriptors: " << descriptors[i] ;

	cout << "locations: " << locations << endl;


	//-------------------------------------------------------------
	//training------------------------http://docs.opencv.org/3.0-beta/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html-----------------
	//convert vector to Mat

	Mat descriptorsT(2, descriptors[0].size(), CV_32FC1);

	Mat test(descriptors[0], true);
	transpose(test, test);
	cout << test.rows << endl;
	cout << test.cols << endl;
	test.row(0).copyTo(descriptorsT.row(0));

	Mat test2(descriptors[0], true);
	transpose(test2, test2);
	test2.row(0).copyTo(descriptorsT.row(1));
	cout << "meh" << endl;
	/*for (int i = 0; i < descriptors[0].size(); i++)
	{
	descriptorsT.at<float>(0, i) = descriptors[0][i];
	}
	for (int i = 0; i < descriptors[0].size(); i++)
	{
	descriptorsT.at<float>(1, i) = descriptors[1][i];
	}
	cout << "meh" << endl;*/
	//labels create
	int labels[2] = { 1,-1 };
	Mat labelsMat(2, 1, CV_32SC1, labels);
	//train data creation
	Ptr<TrainData> tData = TrainData::create(descriptorsT, ROW_SAMPLE, labelsMat);//create it or pass directly to the training part ,train<SVM>(trainingDataMat, ROW_SAMPLE, labelsMat, params);
	Ptr<SVM> svm = SVM::create();
	//set parameters of svn
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	//train the svm
	svm->train(tData);
	Mat sv = svm->getSupportVectors();

	//save it

	svm->save("svm.xml");


	waitKey(0);
	return 0;
}

void getNameOfImages(vector<string>& nameOfImages, string xmlname, int& num)//get name of images
{
	FileStorage fs(xmlname, FileStorage::READ);
	fs["num"] >> num; //read number, change this in xml
	for (int i = 1; i < num + 1; i++)
	{
		nameOfImages.push_back(fs["i" + to_string(i)]); //push them back to vector
		cout << nameOfImages[i - 1] << endl;
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
	fs2["ymin"] >> ymin;
	fs2["ymax"] >> ymax;
	fs2["xmin"] >> xmin;
	fs2["xmax"] >> xmax;
	cout << "xmin is " << xmin << ymin << xmax << ymax << endl;

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