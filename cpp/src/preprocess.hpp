//
// Created by 黄梓浩 on 2025/1/15.
//

#ifndef CPP_PREPROCESS_HPP
#define CPP_PREPROCESS_HPP

#endif //CPP_PREPROCESS_HPP
#include "inference.hpp"



void write2txt(const string &txt_path,const string &label){
    ofstream file(txt_path);

    if (!file.is_open()) {
        std::cerr << "无法打开文件！" << std::endl;
        return;
    }
    file << label<< endl;
    file.close();
}

void spilt_video(const string &video_path,const string &label_output_path,const string &img_output_path,const int&skip) {
    Inference I;
    int total_number = 0;
    for (const auto &entry: fs::directory_iterator(video_path)) {
        if (entry.path().filename() == ".DS_Store") {
            continue;
        }
        int number = 0;
        cout << entry.path().filename()  ;

        cv::VideoCapture cap(entry.path());
        if (!cap.isOpened()) {
            cerr << "Error: Couldn't open the video file." << endl;
            return;
        }

        int totalFrames = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
        string output_video_path = label_output_path + "/" + string(entry.path().filename());
        vector<vector<float>> labels;
        for (int j = 0; j < totalFrames; j++) {
            Mat frame;
            cap >> frame;
            if (frame.empty()) {
                break; // 如果没有帧了，退出循环
            }
            if (j % skip == 0) {
                string img_path = img_output_path + "/" + string(entry.path().filename())+"_"+to_string(j) + ".jpg";
                cv::imwrite(img_path,frame);
                I.get_img(frame);
                I.infer_img();
                tie(frame, labels) = I.nms();
                string txt_path = label_output_path + "/" + string(entry.path().filename())+"_"+to_string(j) + ".txt";
                ostringstream oss;
                if (!labels.empty()){
                    number++;
                }
                for (size_t i = 0; i < labels.size(); ++i) {
                    for (size_t jj = 0; jj < labels[i].size(); ++jj) {
                        oss << labels[i][jj];
                        // 在最后一个元素后不添加空格
                        if (!(i == labels.size() - 1 && jj == labels[i].size() - 1)) {
                            oss << " ";
                        }
                    }
                    if (!oss.str().empty()){
                        write2txt(txt_path, oss.str());
                        oss.str("");
                        oss.clear();

                    }
                }
            }
        }
        cap.release();
        cout<<"        number: "<<number<<endl;
        total_number = total_number+number;
    }
    cout<<"\n"<<"\n"<<"total number: "<<total_number<<endl;
}