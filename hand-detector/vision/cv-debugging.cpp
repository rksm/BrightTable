#include "vision/cv-debugging.hpp"
#include "vision/cv-helper.hpp"
#include <numeric>
#include <algorithm>

namespace cvdbg
{

using namespace cv;

int BOXSIZE = 500;

Point getPt1(Mat frame) {
    return Point(frame.cols,frame.rows - BOXSIZE);
}

Point getPt2(Mat frame) {
    return Point(0,frame.rows);
}

void drawText(string text, Mat frame) {
    int baseline = 0;
    const int fontFace = CV_FONT_HERSHEY_SIMPLEX;
    const double fontScale = 2;
    const int thickness = 1;

    static Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);

    static Point textOrg(
        (getPt1(frame).x - getPt2(frame).x)/2-textSize.width/2,
        (getPt1(frame).y + getPt2(frame).y)/2+textSize.height/2);

    putText(frame, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
}

vector<Mat> recordedImages;

void recordImage(const Mat im, string title = "") {
    Mat resizedIm, recordedIm = im.clone();
    if (recordedIm.type() == CV_8UC1)
        cvtColor(recordedIm, recordedIm, CV_GRAY2RGB);
    cvhelper::resizeToFit(recordedIm, resizedIm, 700, 700);
    drawText(title, resizedIm);
    recordedImages.push_back(resizedIm);
}

Mat combineImages(vector<Mat> images) {

    int width = 0, height = 0;

    for (auto im : images) {
        width = width + im.cols;
        height = max(height, im.rows);
    }

    Mat result = Mat(height, width, CV_8UC4);

    int currentCol = 0;
    for (auto im : images) {
        // if (im.type() == CV_8UC1) cvtColor(im, im, CV_GRAY2RGB);
        Rect roi(Point(currentCol, 0), Size(im.cols, im.rows));
        Mat destinationROI = result(roi);
        im.copyTo(destinationROI);
        currentCol += im.cols;
    }

    return result;
}

Mat getAndClearRecordedImages() {
    Mat result;
    if (!recordedImages.empty()) {
        // std::cout << recordedImages.size() << std::endl;
        result = combineImages(recordedImages);
        recordedImages.clear();
    } else {
        result = Mat(200, 200, CV_8U, Scalar::all(0));
    }
    return result;
}

void saveRecordedImages(const string& filename) {
    imwrite(filename, getAndClearRecordedImages());
}

}
