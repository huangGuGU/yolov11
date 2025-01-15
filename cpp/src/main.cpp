#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include "inference.hpp"

namespace fs = std::__fs::filesystem;
using namespace std;
using cv::Mat;


void camera_detect() {
    Inference I;
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return;
    }
    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Could not grab frame." << std::endl;
            break;
        }
        I.get_img(frame);
        I.infer_img();
        I.nms_and_show();
    }
    cap.release();
    cv::destroyAllWindows();
}


void img_detect(const string &img_path) {
    Inference I;
    for (const auto &entry: fs::directory_iterator(img_path)) {
        if (entry.path().filename() == ".DS_Store") {
            continue;
        }
        I.get_img(entry.path());
        I.infer_img();
        I.nms_and_show();
    }
}

int main() {


    img_detect("/Users/hzh/Desktop/cat_frame");
//    camera_detect();
}







