#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <string>

#include "objectbbox.h"

class Yolo11 {
public:
    using ClassChecker = std::function<bool(int, const std::string&)>; // int, string을 매개변수로 받고 bool 반환하는 함수 타입을 저장하는 함수 객체

private:
    cv::dnn::Net net_;
    cv::Size input_size_;
    std::map<int, std::string> class_names_;
    float min_conf_;
    float iou_thresh_;
    ClassChecker valid_class_checker_;

    cv::Mat preprocess(const cv::Mat& image);
    std::vector<ObjectBBox> postprocess(const cv::Mat& output, const cv::Size& original_size);
    void loadClassNames(const std::string& names_file);

public:
    Yolo11(const std::string& model_path, float min_conf = 0.45f, float iou_thresh = 0.45f, ClassChecker valid_class_checker = nullptr, const std::string& names_file = "");
    std::map<int, std::string> getClassIdNamePairs() const;
    std::vector<ObjectBBox> detect(const cv::Mat& image);
};


