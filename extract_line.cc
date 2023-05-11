#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

#include <vector>
cv::Mat CalculateArea(const cv::Mat& edge_img, const double area_thresold) {
  // 统计边缘包围的面积
  std::vector<std::vector<cv::Point>> contours;
  findContours(edge_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  double totalArea = 0;
  cv::Mat img(edge_img.size(), CV_8UC3, cv::Scalar(0, 0, 0));
  for (int i = 0; i < contours.size(); i++) {
    double area = contourArea(contours[i]);
    std::cout << "area(" << i << "): " << area << std::endl;
    if (area > area_thresold) {
      drawContours(img, contours, i, cv::Scalar(0, 0, 255), 1);
    }
    totalArea += area;
  }

  int maxContourIndex = -1;
  double maxContourLength = 0;
  for (int i = 0; i < contours.size(); i++) {
    double contourLength = arcLength(contours[i], true);
    if (contourLength > maxContourLength) {
      maxContourIndex = i;
      maxContourLength = contourLength;
    }
  }

  // 提取轮廓骨干

  return img;
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
    std::cout << "\ncurrent file name: " << filename << std::endl;

    // 读取图像
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    // 定义核大小和形状
    int kernel_size = 3;
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));

    // 进行开运算
    cv::Mat result_open;
    cv::Mat result_close;
    cv::morphologyEx(img, result_open, cv::MORPH_OPEN, kernel);

    kernel_size = 3;
    kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                       cv::Size(kernel_size, kernel_size));
    cv::morphologyEx(result_open, result_close, cv::MORPH_CLOSE, kernel);

    // 高斯滤波
    cv::Mat img_blur;
    GaussianBlur(img, img_blur, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

    // Canny边缘检测
    cv::Mat edges;
    Canny(img_blur, edges, 50, 150, 3);

    const int h = img.rows;
    const int w = img.cols;

    cv::Mat canvas(h * 2, w * 2, CV_8UC1, cv::Scalar(0, 0, 0));

    cv::Mat roi1 = canvas(cv::Rect(0, 0, w, h));
    img.copyTo(roi1);

    cv::Mat roi2 = canvas(cv::Rect(w, 0, w, h));
    edges.copyTo(roi2);

    cv::Mat roi3 = canvas(cv::Rect(0, h, w, h));
    result_open.copyTo(roi3);

    cv::Mat roi4 = canvas(cv::Rect(w, h, w, h));
    result_close.copyTo(roi4);

    cv::Mat contours = CalculateArea(edges, 60);

    cv::Mat skeleton;
    cv::ximgproc::thinning(img, skeleton, cv::ximgproc::THINNING_ZHANGSUEN);
    cv::imshow("skeleton", skeleton);

    // 显示结果
    cv::imshow("Input Image", img);
    cv::imshow("Edge Image", edges);
    cv::imshow("Opening Result", result_open);
    cv::imshow("Close Result", result_close);
    cv::imshow("All Image", canvas);
    cv::imshow("contour Image", contours);
    cv::waitKey(0);
  }

  return 0;
}
