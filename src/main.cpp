#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

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

	cv::Mat img = cv::imread("data/logo.png");
	cv::namedWindow("OpenCV");
	cv::imshow("OpenCV", img);
	cv::waitKey(0);
	return 0;
}