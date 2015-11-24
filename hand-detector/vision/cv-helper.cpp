#include "vision/cv-helper.hpp"
#include <numeric>
#include <algorithm>

namespace cvhelper
{

using namespace cv;

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// colors

RNG rng(12345);
const Scalar randomColor() {
    return Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
}

void resizeToFit(Mat &in, Mat &out, float maxWidth, float maxHeight)
{
  cv::Size size = in.size();
  if (size.width <= maxWidth && size.height <= maxHeight)
  { if (&in != &out) out = in; return; }

  float h = size.height, w = size.width;
  if (h > maxHeight) {
    w = round(w * (maxHeight / h));
    h = maxHeight;
  }
  if (w > maxWidth) {
    h = round(h * (maxWidth / w));
    w = maxWidth;
  }
  resize(in, out, Size(w,h));
}

}
