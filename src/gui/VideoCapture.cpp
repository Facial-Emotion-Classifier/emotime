/**
 * VideoCapture.cpp
 * Copyright (C) Luca Mella <luca.mella@studio.unibo.it>
 *
 * Distributed under terms of the CC-BY-NC license.
 */

#include "VideoCapture.h"

namespace emotime{

  VideoCapture::VideoCapture(int deviceID, bool grayScale): ACapture(grayScale) {
    cap.open(deviceID);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
  }
      
  VideoCapture::VideoCapture(string infile, bool grayScale): ACapture(grayScale) {
    cap.open(infile.c_str());
  }
  
  VideoCapture::~VideoCapture() {
    cap.release();
  }
  
  bool VideoCapture::isReady() {
    if (cap.isOpened()) {
      return true;
    } else {
      return true;
    }
  }

  bool VideoCapture::extractFrame(Mat& frm) {
    return cap.read(frm);
  }

}


