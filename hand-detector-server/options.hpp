#ifndef HAND_DETECTION_SERVER_OPTIONS_H_
#define HAND_DETECTION_SERVER_OPTIONS_H_


#include <iostream>
#include <string>

#include "vision/hand-detection.hpp"
#include "vision/screen-detection.hpp"
#include "json/json.h"

vision::quad::Options quadOptions(Json::Value&);
vision::screen::Options screenOptions(Json::Value&);
vision::hand::Options handOptions(Json::Value&);


#endif  // HAND_DETECTION_SERVER_OPTIONS_H_