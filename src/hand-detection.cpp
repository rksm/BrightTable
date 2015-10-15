#include "hand-detection.hpp"
#include "timer.hpp"
#include <numeric>
#include <algorithm>

using namespace cv;

using PointV = vector<Point>;

const Scalar RED   = Scalar(0, 255,   0);
const Scalar GREEN = Scalar(0,   255, 0);
const Scalar BLUE  = Scalar(0,   0,   255);
const Scalar WHITE = Scalar(255, 255, 255);
const Scalar BLACK = Scalar(0,   0,   0);

int lowThreshold = 50;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;

// How far apart can convexity defect hull points lay apart to still be
// considered as one fingertip?
const int fingerTipWidth = 100; 

struct ConvexityDefect
{
    Point onHullStart;  // point of the contour where the defect begins
    Point onHullEnd;    // point of the contour where the defect ends
    Point defect;       // the farthest from the convex hull point within the defect
    float distFromHull; // distance between the farthest point and the convex hull
    float distFromCenter;
};

Mat prepareForContourDetection(
    Mat src,
    bool renderDebugImages = false)
{

    if (renderDebugImages) recordImage(src, "orig");

    // Mat dst = Mat::zeros(src.size(), CV_8UC1);
    Mat dst;
    cvtColor(src, dst,CV_RGB2GRAY);

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    // blur(src_gray, dst, Size(4,4));
    // equalizeHist(dst, dst);
    medianBlur(dst, dst, 11);
    if (renderDebugImages) recordImage(dst, "blur");

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    threshold(dst, dst, 150, 255, CV_THRESH_BINARY_INV);
    // threshold(dst, dst, 100, 255, CV_THRESH_BINARY);
    // adaptiveThreshold(dst, dst, 115, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 9, -3);
    // adaptiveThreshold(dst, dst, 165, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 9, -3);
    // if (renderDebugImages) recordImage(dst, "threshold");

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    dilate(dst, dst, Mat(3,3, 0), Point(-1, -1), 5);
    if (renderDebugImages) recordImage(dst, "dilate");

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    /// Canny detector
    // Canny(dst, dst, lowThreshold, lowThreshold*ratio, kernel_size);
    // if (renderDebugImages) recordImage(dst, "canny");


    return dst;
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void contourHullExtraction(
    PointV &contours,
    PointV &hullPoints,
    vector<int> &hullInts, vector<Vec4i> &defects)
{
    convexHull(contours, hullPoints);
    convexHull(contours, hullInts);
    if (hullPoints.size() >= 3) {
        convexityDefects(contours, hullInts, defects);
    }
}

bool contourMoments(PointV &contours, vector<Moments> &momentsVec)
{
    Moments mo = moments(contours, false);
    if (mo.m00 > 0) { // self intersecting...
        // Point pos = Point((int) (mo.m10 / mo.m00), (int) (mo.m01 / mo.m00));
        // std::cout
        //     << "Moments: [" << i << "] "
        //     << mo.m10 << "/" << mo.m00 << " = " << pos.x << ","
        //     << mo.m01 << "/" << mo.m00 << " = " << pos.y
        //     << std::endl;
        momentsVec.push_back(mo);
        return true;
    }
    return false;
}

void contourBounds(PointV &contours, Point &palmCenter, int &palmRadius, RotatedRect &cBounds, RotatedRect &defectBounds)
{
    cBounds = fitEllipse(contours);
    Size size = cBounds.size;
    defectBounds = RotatedRect(
        cBounds.center,
        Size2f(size.width*0.66, size.height*0.66),
        cBounds.angle);
    palmRadius = max(defectBounds.size.width, defectBounds.size.height)/2;
    palmCenter = defectBounds.center;
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void drawHull(Mat &drawing, Scalar color, PointV &hullPoints)
{
    for (auto hullp : hullPoints) {
        circle(drawing, hullp,   4, Scalar(255,255,0), 2 );
    }
}

void drawMoments(Mat &drawing, Scalar color, vector<Moments> momentsVec)
{
    Moments mo = momentsVec.back();
    Point pos = Point((int) (mo.m10 / mo.m00), (int) (mo.m01 / mo.m00));
    circle(drawing, pos, 10, color, 5);
}

vector<ConvexityDefect> convexityDefects(Mat &drawing, PointV &contours, int palmRadius, Point palmCenter, vector<Vec4i> &defects)
{
    vector<ConvexityDefect> result;

    for (Vec4i d : defects) {
        int startidx = d[0], endidx = d[1], faridx = d[2];
        Point onHullStart(contours[startidx]); // point of the contour where the defect begins
        Point onHullEnd(contours[endidx]); // point of the contour where the defect ends
        Point defect(contours[faridx]);// the farthest from the convex hull point within the defect
        float distFromHull = d[3] / 256; // distance between the farthest point and the convex hull
        // if (distFromHull > 20)
        {
          // std::cout << norm(defect - palmCenter) << " vs " << palmRadius << std::endl;
            // if (norm(defect - palmCenter) < palmRadius) {
                ConvexityDefect data {
                  onHullStart,                 // onHullStart
                  onHullEnd,                   // onHullEnd
                  defect,                      // defect
                  distFromHull,                // distFromHull
                  (float)norm(defect - palmCenter) // distFromCenter
                };
                result.push_back(data);
            // }

        }
    }

    return result;
}

void drawConvexityDefects(Mat &drawing, Scalar color, vector<ConvexityDefect> defects)
{
    for (auto d : defects) {
      // line(drawing, ptStart, ptEnd, CV_RGB(255,0,0), 2 );
      line(drawing, d.onHullStart, d.defect, CV_RGB(255,0,0), 6);
      line(drawing, d.onHullEnd, d.defect, CV_RGB(255,0,0), 6);
      // Point mid = (onHullStart+onHullEnd)*.5;
      // line(drawing, mid, defect, CV_RGB(255,0,0), 2 );
      // line(drawing, onHullEnd, defect, color, 2 );
      // circle(drawing, onHullStart,   4, Scalar(100,0,255), 2 );
      circle(drawing, d.defect,   18, CV_RGB(255,0,0), 10 );
    }
}

struct Line {
  Point p1;
  Point p2;
  bool operator==(const Line &l2) { return p1 == l2.p1 && p2 == l2.p2; }
};

struct Triangle {
  Point p1;
  Point p2;
  Point p3;
  bool operator==(const Triangle &t) { return p1 == t.p1 && p2 == t.p2 && p3 == t.p3; }
  bool operator<(const Triangle &t) const { return norm(p1) < norm(t.p1); }
  friend std::ostream& operator<< (std::ostream& o, const Triangle &t)
  {
    return o << "Triangle<" << t.p1 << "," << t.p2 << "," << t.p3 << std::endl;
  }
};


// std::ostream& operator<<(std::ostream &o, const Triangle &t)
// {
//     return o << "Triangle<" << t.p1 << "," << t.p2 << "," << t.p3 << std::endl;
// }

// using Line = std::tuple<Point, Point>;
// using Triangle = std::tuple<Point, Point, Point>;

vector<Finger> findFingerTips(vector<ConvexityDefect> defects, PointV hullPoints, Rect innerBounds)
{
  // Guess 1: if angle from defect point to the two connecting hull points is
  // sharp it is probably a finger
  // Guess 2: if we have one finger tip (hull point) that is close or equal to
  // another finger tip and the lines to their defects are nearly parallel, we
  // have another (pointing) finger

  vector<ConvexityDefect> defectsCopy(defects.size());
  std::rotate_copy(defects.begin(), defects.begin()+1, defects.end(), defectsCopy.begin());
  vector<Finger> result;

  for (auto it = defects.begin(), it2 = defectsCopy.begin(); it != defects.end(); it++, it2++) {
      // std::cout << norm(it->onHullStart - it2->onHullEnd) << "...." << norm(it->onHullEnd - it2->onHullStart) << std::endl;
      if (norm(it->onHullStart - it2->onHullEnd) < fingerTipWidth) {
        Point mid = it->onHullStart + (it2->onHullEnd - it->onHullStart) * .5;        
        Finger f{it->defect, it2->defect, mid};
        // std::cout << radToDeg(f.angle()) << std::endl;
        if (radToDeg(f.angle()) < 89) result.push_back(f);
      }
      if (norm(it->onHullEnd - it2->onHullStart) < fingerTipWidth) {
        Point mid = it->onHullEnd + (it2->onHullStart - it->onHullEnd) * .5;        
        Finger f{it->defect, it2->defect, mid};
        // std::cout << radToDeg(f.angle()) << std::endl;
        if (radToDeg(f.angle()) < 89) result.push_back(f);
      }
  }
  // std::cout << result.size() << std::endl;

  return result;
}

vector<HandData> findContours(
  const Mat src,
  bool renderDebugImages = false,
  Mat projectionTransform = Mat::eye(3,3, CV_32F),
  Size projSize = Size(400,300))
{
    vector<HandData> result;
    vector<Vec4i> hierarchy;
    vector<PointV > contours;
    findContours(src, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1, Point(0, 0));
    vector<PointV> hullsP (contours.size());
    vector<cv::vector<int>> hullsI(contours.size());
    vector<vector<Vec4i> > defects(contours.size());
    vector<vector<Moments>> momentsVec(contours.size());
    Size imageSize = src.size();
    long minArea = ((imageSize.width * imageSize.height) / 100) * 5; // 5%
    int innerImageOffset = 100;
    Rect innerImageBounds {
      Point(innerImageOffset, innerImageOffset),
      Size(imageSize.width-innerImageOffset, imageSize.height-innerImageOffset)};

    Mat debugImage = Mat::zeros(imageSize, CV_8UC3);

    // std::cout << "Found " << contours.size() << " contours" << std::endl;
    /// Draw contours
    for (int i = 0; i< contours.size(); i++)
    {
        double area = std::abs(contourArea(contours[i], true));
        if (area > minArea && contours[i].size() >= 5)
        {

            RotatedRect cBounds, defectBounds; int radius; Point center;
            contourBounds(contours[i], center, radius, cBounds, defectBounds);
            if ((center.x < imageSize.width) && (center.y < imageSize.height)) {
              contourHullExtraction(contours[i], hullsP[i], hullsI[i], defects[i]);
              bool momentsFound = contourMoments(contours[i], momentsVec[i]);
              vector<ConvexityDefect> defectData = convexityDefects(debugImage, contours[i], radius, center, defects[i]);
              vector<Finger> fingers = findFingerTips(defectData, hullsP[i], innerImageBounds);

              result.push_back(HandData{radius,center, cBounds, defectBounds, fingers});

              if (renderDebugImages) {
                auto color = randomColor();
                drawContours(debugImage, contours, i, color, 2, 8, hierarchy, 0, Point());
                drawContours(debugImage, hullsP, i, color, 1, 8, vector<Vec4i>(), 0, Point());
                drawHull(debugImage, color, hullsP[i]);
                if (momentsFound) drawMoments(debugImage, color, momentsVec[i]);
                ellipse(debugImage, cBounds, CV_RGB(255,0,0), 2, 8 );
                ellipse(debugImage, defectBounds, CV_RGB(0,255,0), 3, 8 );
                circle(debugImage, center, radius, CV_RGB(0,0,255), 3, 8 );
                // drawConvexityDefects(debugImage, color, defectData);
                for (auto f : fingers) {
                  line(debugImage, f.base1, f.tip, CV_RGB(255,0,0), 6);
                  line(debugImage, f.base2, f.tip, CV_RGB(255,0,0), 6);
                  circle(debugImage, f.tip,   18, CV_RGB(255,0,0), 10 );
                }
              }
            }

        }
    }

    if (renderDebugImages) recordImage(debugImage, "hand data");
    // cv::warpPerspective(debugImage, debugImage, projectionTransform, projSize);
    // if (renderDebugImages) recordImage(debugImage, "hand data");

    return result;
}


// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

const int HIGH = 150;
const int LOW = 150;
const int maxImageWidth = 1000;
const int maxImageHeight = 1000;

FrameWithHands processFrame(Mat src, bool renderDebugImages, Mat projectionTransform, Size projSize)
{

    vector<HandData> hands;
    std::cout << timeToRunMs([&](){
        Mat resized = resizeToFit(src, maxImageWidth, maxImageHeight);
        Mat thresholded = prepareForContourDetection(resized, renderDebugImages);
        hands = findContours(thresholded, renderDebugImages, projectionTransform, projSize);
    }).count() << std::endl;

    return FrameWithHands {std::time(nullptr), src.size(), hands};

}
