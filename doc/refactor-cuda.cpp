//TODO: Before and After?

#include <opencv2/gpu/gpu.hpp>

....

using cv::gpu::CascadeClassifier_GPU;

FaceDetector::FaceDetector(std::string face_config_file, std::string eye_config_file){
  this->face_config_file = face_config_file;
  this->eye_config_file = eye_config_file;

  cout << "CUDA Device Count: " << getCudaEnabledDeviceCount() << endl;
  
  ....
}

bool FaceDetector::detectFace(cv::Mat& img, cv::Rect& face) {
  cv::gpu::CascadeClassifier_GPU cascade_f2;
  cascade_f2.load(this->face_config_file);
  
  GpuMat gfaces;
  GpuMat gray_gpu(img);
  
  equalizeHist(img, img);

  // Find Faces
  int detect_num = cascade_f2.detectMultiScale(gray_gpu, gfaces, 1.1, 2, this->faceMinSize );

  ....
  
  gfaces.release();
  gray_gpu.release();
  return true;
}

/***Similar changes for detectEyes***/
