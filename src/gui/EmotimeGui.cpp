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

  string emoToPath(Emotion emo, string prefix) {
	  switch (emo) {
		  case NEUTRAL: return string(prefix + "neutral.png");
		  case ANGER: return string(prefix + "anger.png");
		  case CONTEMPT: return string(prefix + "contempt.png");
		  case DISGUST: return string(prefix + "disgust.png");
		  case FEAR: return string(prefix + "fear.png");
		  case HAPPY: return string(prefix + "happy.png");
		  case SADNESS: return string(prefix + "sadness.png");
		  case SURPRISE: return string(prefix + "surprise.png");
		  case OTHERS: return string(prefix + "others.png");
		  default: return string(prefix + "unknown.png");
	  }
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
	string path = emoToPath(prediction.first, "../src/gui/images/");

    emoji = imread(path);
    /*emoji.copyTo(frame.rowRange(150, 250).colRange(20, 120));*/
    resize(emoji, emoji, Size(100, 100), 1);
	cvtColor(emoji, emoji, CV_BGR2GRAY);
    emoji.copyTo(frame (Rect(20, 140, 100, 100)));

    // QT only
    //displayOverlay(mainWinTitle.c_str(), osd.c_str(), 2000);
    return true;
  }


}


