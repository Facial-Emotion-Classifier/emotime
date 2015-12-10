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
    
    ....
    
    Mat emoji;
	string path = emoToPath(prediction.first, "../src/gui/images/");

    emoji = imread(path);
    /*emoji.copyTo(frame.rowRange(150, 250).colRange(20, 120));*/
    resize(emoji, emoji, Size(100, 100), 1);
	cvtColor(emoji, emoji, CV_BGR2GRAY);
    emoji.copyTo(frame (Rect(20, 140, 100, 100)));
