#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

#include <vector>

cv::Mat ProcessImg(const cv::Mat& input) {
  const int kernel_size = 3;
  cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));

  cv::Mat img_open;
  cv::Mat img_close;

  // 进行开运算
  cv::morphologyEx(input, img_open, cv::MORPH_OPEN, kernel);
  // 进行闭运算
  cv::morphologyEx(img_open, img_close, cv::MORPH_CLOSE, kernel);

  cv::Mat skeleton;
  // 提取骨干
  cv::ximgproc::thinning(img_close, skeleton, cv::ximgproc::THINNING_GUOHALL);

  cv::Mat result;
  // 或 
  cv::bitwise_and(input, skeleton, result);
  return result;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    return -1;
  }
  int count = 0;

  while (true) {
    std::stringstream ss;
    ss << argv[1];
    ss << "/";
    ss << count++;
    ss << ".png";
    std::string filename = ss.str();
    FILE* file = std::fopen(filename.c_str(), "r");
    if (file == NULL) {
      break;
    }
    std::fclose(file);
    std::cout << "\rcurrent file name: " << filename << std::flush;

    // 读取图像
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    cv::Mat result = ProcessImg(img);
    // 高斯滤波
    const int h = img.rows;
    const int w = img.cols;

    cv::Mat canvas(h, w * 2, CV_8UC1, cv::Scalar(0, 0, 0));

    cv::Mat roi1 = canvas(cv::Rect(0, 0, w, h));
    img.copyTo(roi1);

    cv::Mat roi2 = canvas(cv::Rect(w, 0, w, h));
    result.copyTo(roi2);

    // 显示结果
    cv::imshow("All Image", canvas);
    cv::waitKey(0);
  }

  return 0;
}
