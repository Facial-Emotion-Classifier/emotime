/**
 *
 * @file    EmotimeGui.cpp
 * @brief   Implementation of EmotimeGUI
 *
 */

#include "EmotimeGui.h"

namespace emotime{

  EmotimeGui::EmotimeGui(FacePreProcessor* fp, EmoDetector* detect, int fps) :
    EmotimeGui::AGui(new WebcamCapture(true), fp, detect, fps, "Emotime!") {

  }

  EmotimeGui::EmotimeGui(ACapture* capture, FacePreProcessor* fp, EmoDetector*
      detect, int fps) : EmotimeGui::AGui(capture, fp, detect, fps,
        "Emotime!") {
   }

  EmotimeGui::~EmotimeGui() {
    //delete this->capture;
  }

  bool EmotimeGui::newFrame(Mat& frame, pair<Emotion, float> prediction) {
    Mat copy;
    frame.copyTo(copy);
    stringstream ss, ss2;
    ss << "Emotion: " << emotionStrings(prediction.first);
    ss2 << "Score: " << prediction.second;
    string osd = ss.str();
    string osd2 = ss2.str();
    
    std::cout << emotionToEmoji(prediction.first) << "\n" << std::flush;

    cv::putText(frame, osd.c_str(), Point(20,60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar::all(255));
    cv::putText(frame, osd2.c_str(), Point(20,100), FONT_HERSHEY_SIMPLEX, 0.7, Scalar::all(255));

    Mat emoji;
    emoji = imread("images/emoji.png");
    /*emoji.copyTo(frame.rowRange(150, 250).colRange(20, 120));*/
    resize(frame, emoji, Size(100, 100), 1);
    emoji.copyTo(frame (Rect(100, 100, 100, 100)));

    // QT only
    //displayOverlay(mainWinTitle.c_str(), osd.c_str(), 2000);
    return true;
  }


}


