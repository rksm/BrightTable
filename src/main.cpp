#include "HandDetection.h"
#include <regex>

using cv::Mat;
using cv::imread;

int main(int argc, char** argv)
{

    Mat frame;

    if (false)
    {
        cv::VideoCapture cap = cv::VideoCapture(0);
        cap.read(frame);
        cv::imwrite("/Users/robert/Lively/LivelyKernel2/opencv-test/test-images/6.png", frame);
    }
    else 
    {
        std::regex reg("\\.[a-z]+$");
        for (auto file : findTestFiles()) {
            auto outFile = std::regex_replace(file, reg, "-result.png");
            frame = cv::imread(file, CV_LOAD_IMAGE_COLOR);
            processFrame(frame);
            saveRecordedImages(outFile);
        }
    }

    return 0;

}
