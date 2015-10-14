#include "HandDetection.hpp"
#include <numeric>
#include <algorithm>

using namespace cv;

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// colors

RNG rng(12345);
const Scalar randomColor() {
    return Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//  rext drawing

int BOXSIZE = 500;

Point getPt1(Mat frame) {
    return Point(frame.cols,frame.rows - BOXSIZE);
}

Point getPt2(Mat frame) {
    return Point(0,frame.rows);
}


// void drawRectangle (Mat frame) {
//     static Point pt1(getPt1(frame));
//     static Point pt2(getPt2(frame));
//     static Scalar white(0,0,0);


//     Mat overlay;
//     frame.copyTo(overlay);

//     rectangle(
//         overlay,
//         pt1,
//         pt2,
//         white,
//         CV_FILLED
//     );
//     double opacity = 0.6;
//     addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame);

// }

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

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// Image recording

vector<Mat> recordedImages;

Mat resizeToFit(Mat mat, int maxWidth, int maxHeight) {
    Mat dst = mat;
    float height = mat.rows, width = mat.cols;
    bool needsResize = false;
    if (height > maxHeight) {
        width = round(width * (maxHeight / height));
        height = maxHeight;
        needsResize = true;
    }
    // std::cout << width << " " << height << std::endl;
    if (width > maxWidth) {
        height = round(height * (maxWidth / width));
        width = maxWidth;
        needsResize = true;
    }
    if (needsResize) {
        dst = Mat(width, height, mat.type());
        resize(mat, dst, Size(width, height));
    }
    return dst;
}

void recordImage(const Mat im, string title = "") {
    Mat recordedIm = im.clone();
    if (recordedIm.type() == CV_8UC1)
        cvtColor(recordedIm, recordedIm, CV_GRAY2RGB);
    resizeToFit(recordedIm, 700, 700);
    drawText(title, recordedIm);
    recordedImages.push_back(recordedIm);
}

Mat combineImages(vector<Mat> images) {

    int width = 0, height = 0;

    for (auto im : images) {
        width = width + im.cols;
        height = max(height, im.rows);
    }

    Mat result = Mat(height, width, CV_8UC3);

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
