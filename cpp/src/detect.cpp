#include "Common.hpp"

void camera_detect() {
    Inference I;
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return;
    }
    cv::Mat frame;
    vector<vector<float>> labels;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Could not grab frame." << std::endl;
            break;
        }
        I.get_img(frame);
        I.infer_img();
        tie(frame,labels) = I.nms();
        cv::imshow("Detections", frame);
        cv::waitKey(1);
    }
    cap.release();
    cv::destroyAllWindows();
}

void img_detect(const string &img_path,const string &save_path) {
    make_dir(save_path);
    Inference I;
    vector<vector<float>> labels;
    cv::Mat frame;

    for (const auto &entry: fs::directory_iterator(img_path)) {
        if (entry.path().filename() == ".DS_Store") {
            continue;
        }
        I.get_img(entry.path());


        I.infer_img();
        tie(frame,labels) = I.nms();
        string save_img_path = save_path+"/"+string(entry.path().filename());
        cv::imwrite( save_img_path,frame);
        cv::imshow("Detections", frame);
        cv::waitKey(1);
        cv::destroyAllWindows();
    }
}

void video_detect(const string &video_path, const string &output_path) {
    make_dir(output_path);
    Inference I;
    for (const auto &entry: fs::directory_iterator(video_path)) {
        if (entry.path().filename() == ".DS_Store") {
            continue;
        }
        cout << entry.path().filename() << " processing...";
        auto start = std::chrono::high_resolution_clock::now();
        cv::VideoCapture cap(entry.path());
        if (!cap.isOpened()) {
            cerr << "Error: Couldn't open the video file." << endl;
            return;
        }
        // 获取视频的帧率、帧数等信息
        int fps = int(cap.get(cv::CAP_PROP_FPS));
        int totalFrames = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
        int width = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int videoLength = totalFrames / fps;
        cout << "           video length: " <<videoLength << " s";

        string output_video_path = output_path + "/" + string(entry.path().filename());
        cv::VideoWriter writer(output_video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                               cv::Size(width, height));
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        vector<vector<float>> labels;
        // 循环读取视频的每一帧，直到达到切割的结束帧
        for (int j = 0; j < totalFrames; ++j) {
            Mat frame;
            cap >> frame;
            if (frame.empty()) {
                break; // 如果没有帧了，退出循环
            }
            I.get_img(frame);
            I.infer_img();
            tie(frame,labels) = I.nms();
            writer.write(frame);
        }
        // 释放资源
        cap.release();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        cout << "           Elapsed time: " << duration.count()/1000 << " s" << endl;
        cout <<string(160, '-')<< endl;
    }
}
