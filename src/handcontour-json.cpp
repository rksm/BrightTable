#include "hand-detection.hpp"
#include "json/json.h"

Json::Value convert(cv::Point data) {
  Json::Value json;
  json["x"] = data.x;
  json["y"] = data.y;
  return json;
}

Json::Value convert(cv::Size data) {
  Json::Value json;
  json["width"] = data.width;
  json["height"] = data.height;
  return json;
}

Json::Value convert(cv::RotatedRect data) {
  Json::Value json;
  json["center"] = convert(data.center);
  json["size"] = convert(data.size);
  json["angle"] = data.angle;
  return json;
}

Json::Value convert(Finger data) {
  Json::Value json;
  json["base1"] = convert(data.base1);
  json["base2"] = convert(data.base2);
  json["tip"] = convert(data.tip);
  return json;
}

Json::Value convert(HandData data) {
  Json::Value json;
  json["palmRadius"] = data.palmRadius;
  json["palmCenter"] = convert(data.palmCenter);
  json["contourBounds"] = convert(data.contourBounds);
  json["fingerTips"] = {};
  for (int i = 0; i < data.fingerTips.size(); i++) {
    json["fingerTips"][i] = convert(data.fingerTips[i]);
  }
  // json["convexityDefectArea"] = data.convexityDefectArea;
  // json["fingerTips"] = data.fingerTips;
  return json;
}

Json::Value convert(FrameWithHands data) {
  Json::Value json;
  json["time"] = (long long)data.time;
  json["imageSize"] = convert(data.imageSize);
  for (int i = 0; i < data.hands.size(); i++) {
    json["hands"][i] = convert(data.hands[i]);
  }
  return json;
}

std::string frameWithHandsToJSONString(FrameWithHands data) {
  Json::FastWriter writer;
  return writer.write(convert(data));
}
