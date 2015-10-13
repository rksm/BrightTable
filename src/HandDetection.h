#ifndef HANDDETECTION_H_
#define HANDDETECTION_H_

#include <stdio.h>
#include <opencv2/opencv.hpp>

using std::string;
using std::vector;
using cv::Mat;
using cv::Point;

const double PI = 3.141592653589793238463;

inline double angleBetween(cv::Point2d p1, cv::Point2d p2, cv::Point2d c = cv::Point(0.0,0.0))
{
  cv::Point2d q1 = p1 - c, q2 = p2 - c;
  return acos((q1 * (1/norm(q1))).ddot(q2 * (1/norm(q2))));
}

inline double radToDeg(double r) { return 180/PI * r; };

struct Finger {
    Point base1;
    Point base2;
    Point tip;
    double angle() const { return angleBetween(base1, base2, tip); };
    bool operator==(const Finger &f) { return base1 == f.base1 && base2 == f.base2 && tip == f.tip; };
    friend std::ostream& operator<< (std::ostream& o, const Finger &f)
    {
      return o << "Finger<"
          << f.base1 << "," << f.base2 << "," << f.tip
          // << radToDeg(f.angle()) << ">" << std::endl;
          << std::endl;
    }

};

struct HandData {
    int palmRadius; // min(width, height) / 2 if contourBounds
    Point palmCenter; // min(width, height) / 2 if contourBounds
    cv::RotatedRect contourBounds;
    cv::RotatedRect convexityDefectArea; // the subset of the contourBounds that we consider for defects
    vector<Finger> fingerTips;
};

struct FrameWithHands {
    std::time_t time;
    cv::Size imageSize;
    vector<HandData> hands;
};

double getOrientation(vector<cv::Point>, Mat);
FrameWithHands processFrame(Mat, bool = false);

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
Mat resizeToFit(Mat, int, int);
const cv::Scalar randomColor();

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
void recordImage(const Mat, string);
void saveRecordedImages(const string&);
Mat getAndClearRecordedImages();

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
vector<string> findTestFiles();

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
string frameWithHandsToJSONString(FrameWithHands data);

#endif  // HANDDETECTION_H_