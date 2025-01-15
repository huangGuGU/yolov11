#ifndef CPP_COMMON_HPP
#define CPP_COMMON_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <tuple>
#include <fstream>
namespace fs = std::__fs::filesystem;
using namespace std;
using cv::Mat;

void camera_detect();
void img_detect(const string &img_path,const string &save_path);
void video_detect(const string &video_path, const string &output_path);


void write2txt(const string &txt_path,const string &label);
void split_video_generate_label(const string &video_path,const string &label_output_path,const string &img_output_path,const int&skip);
void delete_empty_label_file(const string &label_dir_path,const string &img_dir_path,const string &delete_label_path,const string &new_img_path);

void make_dir(const string &dirPath);

class Inference {
public:
    string model_path = "../../best.onnx";
    vector<std::string> classes{"cat"};
    float conf_threshold = 0.45f;
    float nms_threshold = 0.50f;
    float scale;
    int origin_col, origin_row;
    std::vector<int> class_ids;

    cv::Mat formatToSquare(const cv::Mat &source);
    void get_img(const string &img_path);
    void get_img(const cv::Mat &mat);
    void infer_img();
    tuple<cv::Mat, vector<vector<float>>> nms() ;

private:
    vector<cv::Rect> boxes;
    vector<vector<float>> labels;
    vector<float> confidences;
    Mat frame;
};

#endif //CPP_COMMON_HPP