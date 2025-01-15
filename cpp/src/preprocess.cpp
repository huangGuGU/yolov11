#include "Common.hpp"


void make_dir(const string &dirPath){
    if (!fs::exists(dirPath)) {
    fs::create_directory(dirPath);  // 创建目录
    }

}

void write2txt(const string &txt_path,const string &label){
    ofstream file(txt_path);

    if (!file.is_open()) {
        std::cerr << "无法打开文件！" << std::endl;
        return;
    }
    file << label<< endl;
    file.close();
}

void split_video_generate_label(const string &video_path,const string &label_output_path,const string &img_output_path,const int&skip) {
    make_dir(label_output_path);make_dir(img_output_path);
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
                break;
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


void delete_empty_label_file(const string &label_dir_path,const string &img_dir_path,const string &delete_label_path,const string &new_img_path){
    make_dir(delete_label_path);make_dir(new_img_path);

    vector<string> label_list;

    for (const auto& entry : fs::directory_iterator(label_dir_path)) {
        if (entry.is_regular_file()) {
            string filename = entry.path().filename().string();
            if (filename != ".DS_Store" && filename != "classes.txt") {
                label_list.push_back(filename);
            }
        }
    }
    // 遍历标签列表
    for (const auto& label_name : label_list) {
        string label_path = label_dir_path + "/" + label_name;
        string delete_path = delete_label_path + "/" + label_name;

        // 打开标签文件并读取内容
        ifstream file(label_path);
        if (!file.is_open()) {
            cerr << "Failed to open file: " << label_path << std::endl;
            continue;
        }
        string content;
        getline(file, content, '\0'); // 读取整个文件内容
        file.close();

        // 如果内容为空，移动标签文件
        if (content.empty()) {
            fs::rename(label_path, delete_path);
        }
        else {
            // 如果内容不为空，复制图片文件
            string img_path = img_dir_path + "/" + label_name.substr(0, label_name.size() - 3) + "jpg";
            string img_copy_path = new_img_path + "/" + label_name.substr(0, label_name.size() - 3) + "jpg";
            fs::copy(img_path, img_copy_path);
        }
    }
}