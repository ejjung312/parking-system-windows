#include "objectbbox.h"

ObjectBBox::ObjectBBox(const std::string& lbl, int class_id, float conf_, float cx_, float cy_, float w, float h, float scale_x, float scale_y) 
    : label(lbl), class_id(class_id), conf(conf_), cx(cx_), cy(cy_) {
    x1 = (cx_ - w / 2) * scale_x;
    y1 = (cy_ - h / 2) * scale_y;
    x2 = (cx_ + w / 2) * scale_x;
    y2 = (cy_ + h / 2) * scale_y;

    rect = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}

cv::Mat ObjectBBox::draw(cv::Mat& img, cv::Scalar color) const {
    cv::rectangle(img, rect, color, 2);
    cv::putText(img, label + " " + std::to_string(conf).substr(0, 4), cv::Point(rect.x, rect.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

    return img;
}