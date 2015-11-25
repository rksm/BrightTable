#ifndef CV_DEBUGGING_H_
#define CV_DEBUGGING_H_

#include <opencv2/opencv.hpp>

namespace cvdbg
{

void recordImage(const cv::Mat, std::string);
void saveRecordedImages(const std::string&);
cv::Mat getAndClearRecordedImages();

}

#endif  // CV_DEBUGGING_H_