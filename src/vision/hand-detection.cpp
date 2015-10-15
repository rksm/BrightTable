#include "vision/hand-detection.hpp"
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
const int fingerTipWidth = 50; 

template<typename PointType>
PointType meanPos(vector<PointType> points)
{
  PointType c(0,0);
  for (auto p : points) c += p;
  c.x /= points.size();
  c.y /= points.size();
  return c;
}

struct ConvexityDefect
{
  Point onHullStart;  // point of the contour where the defect begins
  Point onHullEnd;    // point of the contour where the defect ends
  Point defect;       // the farthest from the convex hull point within the defect
  float distFromHull; // distance between the farthest point and the convex hull
  float distFromCenter;
};

Mat prepareForContourDetection(
  Mat &src,
  bool renderDebugImages = false)
{

    if (renderDebugImages) recordImage(src, "orig");

    // Mat dst = Mat::zeros(src.size(), CV_8UC1);
    Size size = src.size();
    Mat dst;
    cvtColor(src, dst,CV_RGB2GRAY);

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    // blur(src_gray, dst, Size(4,4));
    // equalizeHist(dst, dst);
    medianBlur(dst, dst, 11);
    // if (renderDebugImages) recordImage(dst, "blur");

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    threshold(dst, dst, 120, 255, CV_THRESH_BINARY_INV);
    // threshold(dst, dst, 100, 255, CV_THRESH_BINARY);
    // adaptiveThreshold(dst, dst, 115, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 9, -3);
    // adaptiveThreshold(dst, dst, 165, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 9, -3);
    // if (renderDebugImages) recordImage(dst, "threshold");

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    dilate(dst, dst, Mat(3,3, 0), Point(-1, -1), 5);

    // "crop" the image...
    int cropWidth = 12;
    auto black = Scalar(0,0,0);
    rectangle(dst, Point(0, 0), Point(cropWidth, size.height), black, CV_FILLED);
    rectangle(dst, Point(0, 0), Point(size.width, cropWidth), black, CV_FILLED);
    rectangle(dst, Point(size.width - cropWidth, 0), Point(size.width, size.height), black, CV_FILLED);
    rectangle(dst, Point(0, size.height), Point(size.width, size.height), black, CV_FILLED);
    if (renderDebugImages) recordImage(dst, "dilate");

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    /// Canny detector
    // Canny(dst, dst, lowThreshold, lowThreshold*ratio, kernel_size);
    // if (renderDebugImages) recordImage(dst, "canny");

    return dst;
}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

HandContour findHandContour(
  PointV contourPoints,
  RotatedRect contourBounds,
  Rect fullImageBounds)
{
  // find which contour points are on the image's edge. This is where the arm
  // starts
  int offset = 15;
  Rect innerRect(offset, offset,
    fullImageBounds.width-2*offset, fullImageBounds.height-2*offset);
  
  PointV pointsOnEdge;
  for (auto p : contourPoints) {
    if (innerRect.contains(p)) continue;
    pointsOnEdge.push_back(p);
  }

  // find the opposite end to the arm start
  Point armStart = minAreaRect(pointsOnEdge).center,
        pointingToP = armStart;
  int maxDist = 0;
  for (auto p : contourPoints)
  {
    int dist = norm(p - armStart);
    if (dist > maxDist) { pointingToP = p; maxDist = dist; }
  }

  // find the bounding box around the palm area. We assume that the hands
  // "width" is around 1/3 of its "length". Mabe more with outstretched fingers?
  float longSide = std::max(contourBounds.size.width, contourBounds.size.height),
        shortSide = std::min(contourBounds.size.width, contourBounds.size.height),
        ratio = (shortSide / longSide),
        longSideShortened = ratio > 0.3 ? longSide : shortSide * 1.6;

  PointV pointsNearCenter;
  for (auto p : contourPoints)
    if (norm(pointingToP - p) <= longSideShortened)
      pointsNearCenter.push_back(p);

  auto boundsAroundHand = minAreaRect(pointsNearCenter);
  
  Point2f to = Point2f(pointingToP.x, pointingToP.y);

  int fingerRadius = max(boundsAroundHand.size.width, boundsAroundHand.size.height)/2;
  Point palmCenter = boundsAroundHand.center + (to - boundsAroundHand.center)*.5;

  return HandContour{
    boundsAroundHand,
    pointsNearCenter,
    armStart, pointingToP,
    fingerRadius,
    palmCenter
  };
}

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

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

void drawRect(Mat &drawing, Scalar color, RotatedRect rect)
{
  Point2f pts[4]; rect.points(pts);
  line(drawing, pts[0], pts[1], color, 6);
  line(drawing, pts[1], pts[2], color, 6);
  line(drawing, pts[2], pts[3], color, 6);
  line(drawing, pts[3], pts[0], color, 6);
}

void drawHull(Mat &drawing, Scalar color, PointV &hullPoints)
{
    for (auto hullp : hullPoints) {
        circle(drawing, hullp,   4, Scalar(255,255,0), 2 );
    }
}

void drawMoments(Mat &drawing, const Scalar color, const vector<Moments> &momentsVec)
{
    const Moments mo = momentsVec.back();
    Point pos = Point((int) (mo.m10 / mo.m00), (int) (mo.m01 / mo.m00));
    circle(drawing, pos, 10, color, 5);
}

vector<ConvexityDefect> convexityDefects(
    Mat &drawing,
    const HandContour c,
    const vector<Vec4i> &defects)
{
    vector<ConvexityDefect> result;

    for (Vec4i d : defects) {
        int startidx = d[0], endidx = d[1], faridx = d[2];
        Point defect(c.contourPoints[faridx]);// the farthest from the convex hull point within the defect
        // if (norm(defect-c.palmCenter) > c.fingerRadius) continue;
        Point onHullStart(c.contourPoints[startidx]); // point of the contour where the defect begins
        Point onHullEnd(c.contourPoints[endidx]); // point of the contour where the defect ends
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
                  (float)norm(defect - c.palmCenter) // distFromCenter
                };
                result.push_back(data);
            // }

        }
    }

    return result;
}

void drawConvexityDefects(
    Mat &drawing,
    const Scalar color,
    const vector<ConvexityDefect> &defects)
{
    for (auto d : defects) {
      // line(drawing, ptStart, ptEnd, CV_RGB(255,0,0), 2 );
      line(drawing, d.onHullStart, d.defect, CV_RGB(0,0,255), 6);
      line(drawing, d.onHullEnd, d.defect, CV_RGB(0,0,255), 6);
      Point mid = (d.onHullStart+d.onHullEnd)*.5;
      line(drawing, mid, d.defect, CV_RGB(255,0,0), 2 );
      // line(drawing, onHullEnd, defect, color, 2 );
      // circle(drawing, onHullStart,   4, Scalar(100,0,255), 2 );
      circle(drawing, d.defect,   15, CV_RGB(255,0,0), 5 );
    }
}

vector<Finger> findFingerTips(
    const vector<ConvexityDefect> &defects,
    const PointV &hullPoints,
    const Rect innerBounds)
{
  // Guess 1: if angle from defect point to the two connecting hull points is
  // sharp it is probably a finger
  // Guess 2: if we have one finger tip (hull point) that is close or equal to
  // another finger tip and the lines to their defects are nearly parallel, we
  // have another (pointing) finger

  Mat debug(innerBounds.size(), CV_8UC3);
  auto color = randomColor();

  if (defects.empty()) return vector<Finger>();

  for (int i = 1; i < defects.size(); i++)
  {
    auto d1 = defects[i], d2 = defects[i-1];
    line(debug, d1.onHullStart, d2.defect, CV_RGB(255,100,0), 3);
    line(debug, d1.onHullStart, d1.defect, CV_RGB(100,255,0), 3);
    std::cout << d1.onHullEnd << d2.onHullStart << std::endl;
    if (norm(d1.onHullEnd - d2.onHullStart) > fingerTipWidth)
    {
      d2.onHullStart = d1.onHullEnd;
    }
  }

  recordImage(debug, "foo");

  // for (int i = 0; i < defects.size(); i++)
  // {
  //   Point before = i > 0 ? defects[i-1].onHullEnd : Point(0,0);
  //   Point after = i < defects.size()-2 ? defects[i+1].onHullStart : Point(0,0);
  // }
  // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
  vector<ConvexityDefect> defectsCopy(defects.size());
  std::rotate_copy(defects.begin(), defects.begin()+1, defects.end(), defectsCopy.begin());
  vector<Finger> result;

  auto it2 = defectsCopy.begin();
  for (auto it = defects.begin(); it != defects.end(); it++, it2++) {
    // std::cout << norm(it->onHullStart - it2->onHullEnd) << "...." << norm(it->onHullEnd - it2->onHullStart) << std::endl;
    if (norm(it->onHullStart - it2->onHullEnd) < fingerTipWidth)
    {
      Point mid = it->onHullStart + (it2->onHullEnd - it->onHullStart) * .5;        
      Finger f{it->defect, it2->defect, mid};
      if (radToDeg(f.angle()) < 89) result.push_back(f);
    }
    if (norm(it->onHullEnd - it2->onHullStart) < fingerTipWidth)
    {
      Point mid = it->onHullEnd + (it2->onHullStart - it->onHullEnd) * .5;        
      Finger f{it->defect, it2->defect, mid};
      if (radToDeg(f.angle()) < 89) result.push_back(f);
    }
  }
  // std::cout << result.size() << std::endl;

  return result;
}

vector<HandData> findContours(const Mat &src, Mat &contourImg, bool renderDebugImages = false)
{
    vector<HandData> result;
    vector<Vec4i> hierarchy;
    vector<PointV > contours;
    findContours(src, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_L1, Point(0, 0));
    vector<PointV> hullsP (contours.size());
    vector<cv::vector<int>> hullsI(contours.size());
    vector<vector<Vec4i> > defects(contours.size());
    vector<vector<Moments>> momentsVec(contours.size());
    Rect imageBounds = Rect(0,0, src.cols, src.rows);
    long minArea = ((imageBounds.width * imageBounds.height) / 100) * 5; // 5%
    int innerImageOffset = 100;
    Rect innerImageBounds {
      Point(innerImageOffset, innerImageOffset),
      Size(imageBounds.width-innerImageOffset, imageBounds.height-innerImageOffset)};

    Mat debugImage = Mat::zeros(imageBounds.size(), CV_8UC3);

    // std::cout << "Found " << contours.size() << " contours" << std::endl;
    /// Draw contours
    for (int i = 0; i< contours.size(); i++)
    {
        double area = std::abs(contourArea(contours[i], true));
        if (contours[i].size() < 5) continue;
        if (area < minArea && contours[i].size() < 500) continue;
        RotatedRect fullContourBounds = fitEllipse(contours[i]);

        if ((fullContourBounds.center.x > imageBounds.width)
         && (fullContourBounds.center.y > imageBounds.height)) continue;

        HandContour handContour = findHandContour(contours[i], fullContourBounds, imageBounds);

        contourHullExtraction(handContour.contourPoints, hullsP[i], hullsI[i], defects[i]);

        bool momentsFound = contourMoments(contours[i], momentsVec[i]);
        vector<ConvexityDefect> defectData = convexityDefects(debugImage, handContour, defects[i]);
        // vector<Finger> fingers = findFingerTips(defectData, hullsP[i], innerImageBounds);

        // FIXME!!!
        vector<Finger> fingers{};
        if (defectData.size() >= 2)
        {
          sort(defectData.begin(), defectData.end(),
            [&](ConvexityDefect a, ConvexityDefect b){
              return norm(a.defect - handContour.pointTowards) < norm(b.defect - handContour.pointTowards);
            });
          fingers.push_back(Finger{defectData[0].defect, defectData[1].defect, handContour.pointTowards});
        }

        result.push_back(HandData{
          handContour.fingerRadius, handContour.palmCenter,
          fullContourBounds, handContour.bounds,
          fingers
        });

        if (renderDebugImages) {
          auto color = randomColor();
          drawContours(debugImage, contours, i, color, 2, 8, hierarchy, 0, Point());
          
          drawRect(debugImage, CV_RGB(0,255,0), handContour.bounds);
          circle(debugImage, handContour.pointTowards, 10, CV_RGB(255,255,255), 3);

          // drawContours(debugImage, hullsP, i, color, 1, 8, vector<Vec4i>(), 0, Point());
          // drawHull(debugImage, color, hullsP[i]);
          // if (momentsFound) drawMoments(debugImage, color, momentsVec[i]);
          // ellipse(debugImage, cBounds, CV_RGB(255,0,0), 2, 8 );
          // ellipse(debugImage, defectBounds, CV_RGB(0,255,0), 3, 8 );
          circle(debugImage, handContour.palmCenter, handContour.fingerRadius, CV_RGB(0,0,255), 3, 8 );

          for (auto d : defectData)
          {
            // circle(debugImage, d.defect, 7, CV_RGB(255,255,0), 2);
            // circle(debugImage, d.onHullStart, 7, CV_RGB(255,100,0), 4);
            // circle(debugImage, d.onHullEnd, 7, CV_RGB(100,255,0), 4);
          }
          // drawConvexityDefects(debugImage, color, defectData);
          for (auto f : fingers) {
            line(debugImage, f.base1, f.tip, CV_RGB(0, 255,0), 3);
            line(debugImage, f.base2, f.tip, CV_RGB(0, 255,0), 3);
            circle(debugImage, f.tip,   15, CV_RGB(255,0,0), 4);
          }
        }

    }

    if (renderDebugImages) {
      recordImage(debugImage, "hand data");
      debugImage.copyTo(contourImg);
    }

    return result;
}


// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

const int HIGH = 150;
const int LOW = 150;
const int maxImageWidth = 1000;
const int maxImageHeight = 1000;

FrameWithHands processFrame(Mat &src, Mat &out, bool renderDebugImages)
{
  Mat resized = resizeToFit(src, maxImageWidth, maxImageHeight);
  Mat thresholded = prepareForContourDetection(resized, renderDebugImages);
  vector<HandData> hands = findContours(thresholded, out, renderDebugImages);
  return FrameWithHands {std::time(nullptr), src.size(), hands};
}
