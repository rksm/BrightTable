#ifndef HANDDETECTION_H_
#define HANDDETECTION_H_

#include <stdio.h>
#include <opencv2/opencv.hpp>

using std::string;
using std::vector;
using cv::Mat;

double getOrientation(vector<cv::Point>, Mat);
Mat processFrame(Mat);

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
Mat resizeToFit(Mat, int, int);
const cv::Scalar randomColor();

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
void recordImage(const Mat, string);
void saveRecordedImages(const string&);

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
vector<string> findTestFiles();

#endif  // HANDDETECTION_H_