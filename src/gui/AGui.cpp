/**
 *
 * @file    AGui.cpp
 * @brief   Contains the implementation for AGui
 *
 */
#include "timer.h"
#include "AGui.h"
#include <iostream>

using namespace std;

volatile int quit_signal=0;
#ifdef __unix__
#include <signal.h>
  extern "C" void quit_signal_handler(int signum) {
    if (quit_signal!=0) exit(0); // just exit already
      quit_signal=1;
    cout << "Will quit at next camera frame (repeat to kill now)\n";
}
#endif

namespace emotime{

  AGui::AGui(ACapture* capt, FacePreProcessor* fp, EmoDetector* detect, int fps, string title) {
       capture = capt;
       preprocessor = fp;
       detector = detect;
       mainWinTitle = title;
       this->fps = fps;
     }

  bool AGui::run() {
    if(!init()) {
      return false;
    }
    #ifdef __unix__
      signal(SIGINT,quit_signal_handler); // listen for ctrl-C
    #endif
    double t0 = timestamp();
    int numFrames = 0;
    while (nextFrame()) {
      if (numFrames >= 1000) {
	numFrames = 0;
	t0 = timestamp();
      }
      numFrames++;
      cout << "FPS: " << numFrames/(timestamp()-t0) << endl;
      if (quit_signal) exit(0);
      int key;
      if (fps <= 0) {
        key = waitKey(0);
      } else {
        key = waitKey((int) 1000.0 / fps);
      }
      if((key & 0xFF) == 27) {
        break;
      }
    }
    return true;
  }

  bool AGui::init() {
    namedWindow(mainWinTitle.c_str(), WINDOW_AUTOSIZE);
    return true;
  }

  bool AGui::nextFrame() {
    Mat frame;
    Mat featvector;
    if (capture->nextFrame(frame)) {
      if (preprocessor->preprocess(frame, featvector)) {
        vector<pair<Emotion, float>> predictions = detector->predict(featvector);
        if (!newFrame(frame, predictions)) {
          return false;
        }
        imshow(mainWinTitle.c_str(), frame);
      }
      return true;
    } else {
      return false;
    }
  }

}

