#include <iostream>
#include <string>

#include "vision/hand-detection.hpp"
#include "vision/screen-detection.hpp"
#include "json/json.h"

using std::string;
using std::vector;
using Json::Value;


int thresholdType(std::string name)
{
  if      (name == "CV_THRESH_BINARY")     return CV_THRESH_BINARY;
  else if (name == "CV_THRESH_BINARY_INV") return CV_THRESH_BINARY_INV;
  else if (name == "CV_THRESH_TRUNC")      return CV_THRESH_TRUNC;
  else if (name == "CV_THRESH_TOZERO")     return CV_THRESH_TOZERO;
  else if (name == "CV_THRESH_TOZERO_INV") return CV_THRESH_TOZERO_INV;
  else if (name == "CV_THRESH_MASK")       return CV_THRESH_MASK;
  else if (name == "CV_THRESH_OTSU")       return CV_THRESH_OTSU;
  else                                     return CV_THRESH_BINARY;
}

vision::quad::Options quadOptions(Value &data)
{
  vision::quad::Options opts;
  if (data.isMember("debug"))                       opts.debug                       = data["debug"].asBool();
  if (data.isMember("minAngleOfIntersectingLines")) opts.minAngleOfIntersectingLines = data["minAngleOfIntersectingLines"].asFloat();
  if (data.isMember("maxAngleOfIntersectingLines")) opts.maxAngleOfIntersectingLines = data["maxAngleOfIntersectingLines"].asFloat();
  return opts;
}

vision::screen::Options screenOptions(Value &data)
{
  vision::screen::Options opts;
  if (data.isMember("debug"))              opts.debug              = data["debug"].asBool();
  if (data.isMember("blurIntensity"))      opts.blurIntensity      = data["blurIntensity"].asInt();
  if (data.isMember("minThreshold"))       opts.minThreshold       = data["minThreshold"].asInt();
  if (data.isMember("maxThreshold"))       opts.maxThreshold       = data["maxThreshold"].asInt();
  if (data.isMember("thresholdType"))      opts.thresholdType      = thresholdType(data["thresholdType"].asString());
  if (data.isMember("houghRho"))           opts.houghRho           = data["houghRho"].asInt();
  if (data.isMember("houghTheta"))         opts.houghTheta         = data["houghTheta"].asInt();
  if (data.isMember("houghMinLineLength")) opts.houghMinLineLength = data["houghMinLineLength"].asInt();
  if (data.isMember("houghMinLineGap"))    opts.houghMinLineGap    = data["houghMinLineGap"].asInt();
  if (data.isMember("quadOptions")) {
    opts.quadOptions = quadOptions(data["quadOptions"]);
  }
  return opts;
}

vision::hand::Options handOptions(Value &data)
{
  vision::hand::Options opts;
  if (data.isMember("debug"))                     opts.debug                     = data["debug"].asBool();
  if (data.isMember("renderDebugImages"))         opts.renderDebugImages         = data["renderDebugImages"].asBool();
  if (data.isMember("fingerTipWidth"))            opts.fingerTipWidth            = data["fingerTipWidth"].asInt();
  if (data.isMember("minHandAreaInPercent"))      opts.minHandAreaInPercent      = data["minHandAreaInPercent"].asFloat();
  if (data.isMember("depthSamplingKernelLength")) opts.depthSamplingKernelLength = data["depthSamplingKernelLength"].asInt();
  if (data.isMember("blurIntensity"))             opts.blurIntensity             = data["blurIntensity"].asInt();
  if (data.isMember("thresholdMin"))              opts.thresholdMin              = data["thresholdMin"].asInt();
  if (data.isMember("thresholdMax"))              opts.thresholdMax              = data["thresholdMax"].asInt();
  if (data.isMember("thresholdType"))             opts.thresholdType             = thresholdType(data["thresholdType"].asString());
  if (data.isMember("dilateIterations"))          opts.dilateIterations          = data["dilateIterations"].asInt();
  if (data.isMember("cropWidth"))                 opts.cropWidth                 = data["cropWidth"].asInt();
  return opts;
}
