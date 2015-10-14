#ifndef SCREEN_DETECTION_H_
#define SCREEN_DETECTION_H_

#include <opencv2/opencv.hpp>

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// screen detection
cv::Mat extractLargestRectangle(cv::Mat, cv::Size, bool = false);
cv::Mat screenProjection(cv::Mat, cv::Size, bool = false);

#endif  // SCREEN_DETECTION_H_