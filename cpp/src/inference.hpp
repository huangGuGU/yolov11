#pragma once
#ifndef YOLOV11_INFERENCE_HPP
#define YOLOV11_INFERENCE_HPP
#endif //YOLOV11_INFERENCE_HPP

#include"Common.hpp"


class Inference {
public:
    string model_path = "../../best.onnx";
    vector<std::string> classes{"cat"};
    float conf_threshold = 0.45f;  // confidence threshold
    float nms_threshold = 0.50f;   // NMS threshold
    float scale;
    int origin_col, origin_row;
    std::vector<int> class_ids;

    cv::Mat formatToSquare(const cv::Mat &source) {
        origin_col = source.cols;
        origin_row = source.rows;

        int _max = MAX(origin_col, origin_row);
        cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
        source.copyTo(result(cv::Rect(0, 0, origin_col, origin_row)));
        return result;
    }

    void get_img(const string &img_path) {
        this->frame = cv::imread(img_path);
        boxes.clear();
        confidences.clear();
    }

    void get_img(const cv::Mat &mat) {
        this->frame = mat.clone();
        boxes.clear();
        confidences.clear();
    }

    void infer_img() {
        cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV); // 使用 OpenCV 的 DNN 后端
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);


        Mat frame_Square = formatToSquare(frame);
        scale = float(frame_Square.cols / 640.0);

        cv::Mat input;
        cv::dnn::blobFromImage(frame_Square, input, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        net.setInput(input);


        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        int dimensions = outputs[0].size[1];
        int rows = outputs[0].size[2];
        //outputs[0] 1*5*8400
        outputs[0] = outputs[0].reshape(1, dimensions);  //5*8400
        cv::transpose(outputs[0], outputs[0]);// 8400*5
        auto *data = (float *) outputs[0].data;  // 8400*5



        for (int i = 0; i < rows; ++i) {
            float classes_scores = data[i * 5 + 4];

            cv::Mat scores(1, int(classes.size()), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;

            minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &class_id);

            if (maxClassScore > conf_threshold) {
                confidences.push_back(float(maxClassScore));
                class_ids.push_back(class_id.x);

                float x = data[i * 5 + 0];
                float y = data[i * 5 + 1];
                float w = data[i * 5 + 2];
                float h = data[i * 5 + 3];

                int left = int((x - 0.5 * w) * scale);
                int top = int((y - 0.5 * h) * scale);
                int width = int(w * scale);
                int height = int(h * scale);

                boxes.emplace_back(left, top, width, height);
            }
        }
    }

    tuple<cv::Mat, vector<vector<float>>> nms() {
        vector<cv::Rect> final_boxes;
        final_boxes.clear();
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);
        for (int idx: nms_result) {

            float class_id = class_ids[idx];
            cv::Rect box = boxes[idx];
            final_boxes.emplace_back(box);
            float x, y, w, h;
            w = float(box.width)/ scale;
            h = float(box.height)/ scale;
            x = float(float(box.x)/scale + 0.5 * w) / origin_col *scale;
            y = float(float(box.y)/scale + 0.5 * h) / origin_row *scale;
            w = w / origin_col  *scale;
            h = h / origin_row  *scale;

            labels.emplace_back(vector<float>{class_id, x, y, w, h});
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 3); // 绘制矩形框
            float alpha = 0.2;
            cv::Mat overlay;

            frame.copyTo(overlay);
            cv::rectangle(overlay, box, cv::Scalar(0, 255, 0), -1);
            cv::addWeighted(overlay, alpha, frame, 1.0 - alpha / 255.0, 0, frame);
        }
        if (nms_result.empty()){
            return make_tuple(frame, vector<std::vector<float>>{});
        }
        return make_tuple(frame, labels);

    }

private:

    vector<cv::Rect> boxes;
    vector<vector<float>> labels;
    vector<float> confidences;
    Mat frame;
};