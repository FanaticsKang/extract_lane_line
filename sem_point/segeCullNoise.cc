#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// #include <pcl/filters/uniform_sampling.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

typedef struct {
  int area_id;
  int area_size;

  int line_a;
  int line_b;
  int line_c;

  Point2i cm;

  Point2i p_1;

  Point2i p_2;

  bool left_is_true;
  bool is_line;

} single_area;

cv::Mat imRead(const string& im_path) {
  cv::Mat im = cv::imread(im_path);
  if (im.data == nullptr) {
    cout << "image err: " << im_path << endl;
    exit(0);
  }
  cv::cvtColor(im, im, cv::COLOR_RGB2GRAY);
  return im.clone();
}

void imWrite(const string& dir_path, const int& im_id, const cv::Mat& im) {
  string path = dir_path + "/" + to_string(im_id) + ".png";
  cv::imwrite(path, im);
  return;
}

void print_debug(const single_area& val) {
  cout << " ------------------- " << endl;
  cout << "area id: " << val.area_id << " size: " << val.area_size << endl;
  cout << "a_b_c: " << val.line_a << " " << val.line_b << " " << val.line_c
       << endl;

  // cout << "x,y,w,h: " << val.area_x << " " << val.area_y << " " << val.area_w
  //      << " " << val.area_h << endl;
  cout << "centroids x y: " << val.cm.x << " " << val.cm.y << endl;

  cout << "p1 x y: " << val.p_1.x << " " << val.p_1.y << endl;
  cout << "p2 x y: " << val.p_2.x << " " << val.p_2.y << endl;
  cout << " ------------------- " << endl;
  return;
}

bool Compare(single_area& val_1, single_area& val_2) {
  return val_1.area_size > val_2.area_size;
}

bool is_corner_point(const int x, const int y, const cv::Mat& im) {
  for (int row = y - 2; row <= y + 2; row++) {
    for (int col = x - 2; col <= x + 2; col++) {
      if (im.at<uchar>(row, col) > 200) {
        cout << "corner_point x: " << col << " y : " << row
             << " pixel: " << int(im.at<uchar>(row, col)) << endl;
        return true;
      }
    }
  }
  return false;
}

bool is_line(const single_area& val) {
  float theta = atan2(val.p_2.y - val.cm.y, val.p_2.x - val.cm.x) -
                atan2(val.cm.y - val.p_1.y, val.cm.x - val.p_1.x);
  if (theta > CV_PI) theta -= 2 * CV_PI;
  if (theta < -CV_PI) theta += 2 * CV_PI;

  theta = theta * 180.0 / CV_PI;
  cout << "theta: " << theta << endl;

  return (theta > 15 || theta < -15) ? false : true;
}

bool get_line_fun(single_area& val) {
  int x2 = val.cm.x;
  int y2 = val.cm.y;

  int x1 = val.p_1.x;
  int y1 = val.p_1.y;

  val.line_a = y2 - y1;
  val.line_b = x1 - x2;
  val.line_c = x2 * y1 - x1 * y2;

  cout << " x1: " << x1 << " y1: " << y1 << " x2: " << x2 << " y2: " << y2
       << endl;

  cout << "a_b_c" << val.line_a << " " << val.line_b << " " << val.line_c
       << endl;
}

cv::Mat image_opening(const cv::Mat& sege_im) {
  cv::Mat im = sege_im.clone();
  cv::Mat im_open;
  cv::Mat im_close;
  cv::Mat open_close;

  // Structuring Elemen
  int elem_size = 3;
  cv::Mat struct_elem =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(elem_size, 2));

  // MORPH_OPEN and MORPH_CLOSE
  cv::morphologyEx(im, im_open, cv::MORPH_OPEN, struct_elem);
  cv::morphologyEx(im, im_close, cv::MORPH_CLOSE, struct_elem);
  cv::morphologyEx(im_open, open_close, cv::MORPH_CLOSE, struct_elem);

  cv::imshow("raw", im);
  cv::imshow("im_open", im_open);
  cv::imshow("im_close", im_close);
  cv::imshow("open_close", open_close);

  // cv::waitKey();

  return im_close.clone();
}

cv::Mat connectedComponent(const cv::Mat& sege_im) {
  int cnt_area;
  cv::Mat src_img, img_bool, labels, stats, centroids, img_color, img_gray;
  src_img = sege_im.clone();
  // src_img = image_opening(sege_im);

  cv::threshold(src_img, img_bool, 0, 255, cv::THRESH_OTSU);

  cnt_area =
      cv::connectedComponentsWithStats(img_bool, labels, stats, centroids);

  cout << "connected Components number: " << cnt_area << endl;

  // 0.0.0 is background color, 255.255.255 is valid landmark
  vector<cv::Vec3b> colors(cnt_area);

  //
  colors[0] = cv::Vec3b(0, 0, 0);

  vector<single_area> l_area, r_area;

  cout << "area 0: " << stats.at<int>(0, cv::CC_STAT_AREA) << endl;

  // case0: mianji filter, culling small area
  for (int i = 1; i < cnt_area; i++) {
    colors[i] = cv::Vec3b(255, 255, 255);
    if (stats.at<int>(i, cv::CC_STAT_AREA) < 100) {
      colors[i] = cv::Vec3b(0, 0, 0);
    } else {
      single_area single_area_1;
      single_area_1.line_a = 0;
      single_area_1.line_b = 0;
      single_area_1.line_c = 0;
      single_area_1.p_2.x = 0;
      single_area_1.p_2.y = 0;
      single_area_1.is_line = false;

      int x = stats.at<int>(i, CC_STAT_LEFT);
      int y = stats.at<int>(i, CC_STAT_TOP);
      int w = stats.at<int>(i, CC_STAT_WIDTH);
      int h = stats.at<int>(i, CC_STAT_HEIGHT);
      int area = stats.at<int>(i, CC_STAT_AREA);

      cout << "x: " << x << " y: " << y << " w: " << w << " h:" << h << endl;

      cout << "area: " << i << " " << area << endl;

      Point2i l_top, l_low, r_top, r_low;
      l_top.x = x;
      l_top.y = y;

      l_low.x = x;
      l_low.y = y + h;

      r_top.x = x + w;
      r_top.y = y;

      r_low.x = x + w;
      r_low.y = y + h;

      bool r_flag = (is_corner_point(l_top.x, l_top.y, src_img) &&
                     is_corner_point(r_low.x, r_low.y, src_img));

      bool l_flag = (is_corner_point(l_low.x, l_low.y, src_img) &&
                     is_corner_point(r_top.x, r_top.y, src_img));

      single_area_1.area_id = i;
      single_area_1.area_size = area;
      single_area_1.cm.x = centroids.at<double>(i, 0);
      single_area_1.cm.y = centroids.at<double>(i, 1);

      if (r_flag == true && single_area_1.cm.x > 230) {
        cout << "r" << endl;
        single_area_1.p_1.x = l_top.x;
        single_area_1.p_1.y = l_top.y;
        single_area_1.p_2.x = r_low.x;
        single_area_1.p_2.y = r_low.y;
        single_area_1.left_is_true = false;
      } else if (l_flag == true && single_area_1.cm.x < 250) {
        cout << "l" << endl;
        single_area_1.p_1.x = l_low.x;
        single_area_1.p_1.y = l_low.y;
        single_area_1.p_2.x = r_top.x;
        single_area_1.p_2.y = r_top.y;
        single_area_1.left_is_true = true;
      } else {
        cout << "null" << endl;
        single_area_1.is_line = false;

        if (single_area_1.cm.x > 230) {
          r_area.push_back(single_area_1);
        } else if (single_area_1.cm.x < 250) {
          l_area.push_back(single_area_1);
        } else {
          ;
        }
      }

      // use the distance of point to line, can culling noise of line
      if (r_flag || l_flag) {
        if (is_line(single_area_1)) {
          single_area_1.is_line = true;
          get_line_fun(single_area_1);
        }
      }

      print_debug(single_area_1);

      if (r_flag) {
        r_area.push_back(single_area_1);
      }
      if (l_flag) {
        l_area.push_back(single_area_1);
      }
      cout << "\n\n\n" << endl;
    }
  }

  cout << "l_area.size: " << l_area.size() << endl;
  cout << "r_area.size:  " << r_area.size() << endl;

  if (l_area.size() >= 2) {
    cout << "left area " << endl;
    sort(l_area.begin(), l_area.end(), Compare);

    // debug
    for (auto it = l_area.begin(); it != l_area.end(); it++) {
      cout << "id： " << it->area_id << " size： " << it->area_size
           << " centroids x " << it->cm.x << endl;
    }

    for (int i = 1; i < l_area.size(); i++) {
      if ((l_area[0].cm.x - l_area[i].cm.x) > -5 &&
          l_area[0].cm.x - l_area[i].cm.x < 80) {
        cout << "give up: id: " << l_area[i].area_id << endl;
        cout << "cm x " << l_area[0].cm.x << " area i: x " << l_area[i].cm.x
             << endl;
        colors[l_area[i].area_id] = cv::Vec3b(0, 0, 0);
        auto it = l_area.begin() + i;
        l_area.erase(it);
      }
    }
  }

  if (r_area.size() >= 2) {
    cout << "right area " << endl;
    sort(r_area.begin(), r_area.end(), Compare);

    for (auto it = r_area.begin(); it != r_area.end(); it++) {
      cout << "id： " << it->area_id << " size： " << it->area_size
           << " centroids x " << it->cm.x << endl;
    }

    for (int i = 1; i < r_area.size(); i++) {
      if ((r_area[i].cm.x - r_area[0].cm.x) > -5 &&
          (r_area[i].cm.x - r_area[0].cm.x) < 80) {
        cout << "give up: id: " << r_area[i].area_id << endl;
        cout << "cm x " << r_area[0].cm.x << " area i: x " << r_area[i].cm.x
             << endl;
        colors[r_area[i].area_id] = cv::Vec3b(0, 0, 0);
        auto it = r_area.begin() + i;
        r_area.erase(it);
      }
    }
  }

  if (r_area.size() >= 2 && r_area[0].line_a == 0) {
    cout << "r not line and size max: " << endl;
    colors[r_area[0].area_id] = cv::Vec3b(0, 0, 0);
    r_area.erase(r_area.begin());
  }

  if (l_area.size() >= 2 && l_area[0].line_a == 0) {
    cout << "l not line and size max: " << endl;
    colors[l_area[0].area_id] = cv::Vec3b(0, 0, 0);
    l_area.erase(l_area.begin());
  }

  cout << "l_area.size: " << l_area.size() << endl;
  cout << "r_area.size:  " << r_area.size() << endl;

  img_color = cv::Mat::zeros(src_img.size(), CV_8UC3);
  for (int y = 0; y < img_color.rows; y++)
    for (int x = 0; x < img_color.cols; x++) {
      int label = labels.at<int>(y, x);
      CV_Assert(0 <= label && label <= cnt_area);
      img_color.at<cv::Vec3b>(y, x) = colors[label];
    }

  cv::imshow("connect", img_color);
  cv::imshow("raw", sege_im);
  cv::waitKey(0);
  return img_color.clone();
}

int main(int argc, char* argv[]) {
  const string im_path = argv[1];

  for (int i = 1; i < 1646; i++) {
    cout << "im id: " << i << endl;
    string path = im_path + "/" + to_string(i) + ".png";
    cv::Mat im = imRead(path);
    // image_opening(im);
    cv::Mat new_im = connectedComponent(im);
    // imWrite("/home/lang/data_set/new/", i, new_im);
  }
  return 0;
}