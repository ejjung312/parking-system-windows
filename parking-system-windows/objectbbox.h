#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class ObjectBBox {
public:
    std::string label;
    int class_id;
    float conf;
    cv::Rect rect;
    float cx, cy;
    float x1, x2, y1, y2;

    ObjectBBox(const std::string& lbl, int class_id, float conf,
        float cx, float cy, float w, float h, float scale_x, float scale_y);
    cv::Mat draw(cv::Mat& img, cv::Scalar color) const; // 클래스 멤버 변수 수정불가
};