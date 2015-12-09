Mat GaborBank::filterImage(cv::Mat & src, cv::Size & featSize) {

  ....
  
  Size bankSize=this->getFilteredImgSize(featSize);
  Mat dest = Mat::zeros(bankSize.height, bankSize.width, type);
  Mat tmp_dest = Mat::zeros(bankSize.height, bankSize.width, type);
  Mat image; src.convertTo(image,type);
  resize(image, image, featSize, CV_INTER_AREA);
  double t0 = timestamp();
  #pragma omp parallel for
  for (unsigned int k = 0; k < bank.size(); k++) {
    emotime::GaborKernel * gk = bank.at(k);
    
    ....
    
    filter2D(image, fimag, CV_32F, imag);
    
    ....
    
    Mat scaled = magn;
    for (unsigned int i = 0; i<(unsigned int) featSize.height; i++) {
      for (unsigned int j = 0; j<(unsigned int)featSize.width; j++) {
        if (type == CV_32F){
          dest.at<float>(i + (k * featSize.height), j) = scaled.at<float>(i,j);
        } else if (type == CV_8U){
          dest.at<uint8_t>(i + (k * featSize.height), j) = scaled.at<uint8_t>(i,j);
        }
      }
    }
    cv::max(scaled, tmp_dest, tmp_dest);
  }
