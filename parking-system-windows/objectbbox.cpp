#include "objectbbox.h"

ObjectBBox::ObjectBBox(const std::string& lbl, int class_id, float conf_, float cx, float cy, float w, float h, float scale_x, float scale_y) : label(lbl), class_id(class_id), conf(conf_) {
    x1 = (cx - w / 2) * scale_x;
    y1 = (cy - h / 2) * scale_y;
    x2 = (cx + w / 2) * scale_x;
    y2 = (cy + h / 2) * scale_y;

    rect = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
}

cv::Mat ObjectBBox::draw(cv::Mat& img) const {
    cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);
    cv::putText(img, label + " " + std::to_string(conf).substr(0, 4),
        cv::Point(rect.x, rect.y - 5),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

    return img;
}