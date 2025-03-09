#include "yolo.h"
#include "debug.h"

#include <fstream>

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

/**
 * IoU - 두 개의 영역(예측된 영역과 실제 영역)의 겹치는 정도를 나타내며, 값이 클수록 정확도가 높음
 */
float calculateIoU(const ObjectBBox& box1, const ObjectBBox& box2) {
    auto intersection = box1.rect & box2.rect;
    if (intersection.empty()) return 0.0f;

    float intersection_area = intersection.area();
    float union_area = box1.rect.area() + box2.rect.area() - intersection_area;

    return intersection_area / union_area;
}

void Yolo11::loadClassNames(const std::string& names_file) {
    std::ifstream file(names_file);
    std::cout << "file name: " << names_file << std::endl;
    assert(file.is_open() && ("Failed to open class names file " + names_file).c_str());

    std::string line;
    int class_id = 0;

    while (std::getline(file, line)) {
        // trim
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

        if (!line.empty()) {
            class_names_[class_id++] = line;
        }
    }

    assert(!class_names_.empty() && "No class names loaded from file `coco.names`");
    DEBUG_PRINT("Loaded" << class_names_.size() << " class names");
}

Yolo11::Yolo11(const std::string& model_path, float min_conf, float iou_thresh, ClassChecker valid_class_checker, const std::string& names_file) : min_conf_(min_conf), iou_thresh_(iou_thresh) {
    net_ = cv::dnn::readNetFromONNX(model_path);
#if defined(CUDA_ACC)
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    std::cout << "Using CUDA" << std::endl;
#elif defined(OPENCL_ACC)
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    std::cout << "Using OPENCL" << std::endl;
#else
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout << "Using CPU" << std::endl;
#endif
    assert(!net_.empty() && "Failed to load ONNX model");

    // 클래스 파일
    std::string resolved_names_file = names_file;
    if (resolved_names_file.empty()) {
        size_t last_slash_idx = model_path.find_last_of("\\/");
        if (std::string::npos != last_slash_idx) {
            resolved_names_file = model_path.substr(0, last_slash_idx + 1) + "coco.names";
        }
        else {
            resolved_names_file = "coco.names";
        }
    }

    input_size_.width = 640;
    input_size_.height = 640;

    loadClassNames(resolved_names_file);

    // 삼항연산자
    valid_class_checker_ = valid_class_checker ? valid_class_checker : [](int, const std::string&) { return true; };
}

cv::Mat Yolo11::preprocess(const cv::Mat& image) {
    cv::Mat resized;
    cv::resize(image, resized, input_size_);
    DEBUG_PRINT_MAT_SHAPE(resized);

    cv::Mat blob;
    cv::dnn::blobFromImage(resized, blob, 1.0 / 255.0, input_size_, 
        cv::Scalar(), true, false, CV_32F);
    DEBUG_PRINT_MAT_SHAPE(blob);

    return blob;
}

std::vector<ObjectBBox> Yolo11::postprocess(const cv::Mat& output, const cv::Size& original_size) {
    DEBUG_PRINT_MAT_SHAPE(output);
    assert(output.dims == 2 && output.cols > 0 &&
        output.rows == (4 + class_names_.size()) &&
        "Invalid output shape");

    std::vector<ObjectBBox> valid_boxes;
    std::vector<bool> suppressed(output.cols, false);
    cv::Point2f scale(
        static_cast<float>(original_size.width) / input_size_.width,
        static_cast<float>(original_size.height) / input_size_.height
    );

    // 최대 confidence 찾기
    std::vector<std::pair<float, int>> conf_idx_pairs;
    for (int i = 0; i < output.cols; ++i) {
        cv::Mat scores = output.rowRange(4, output.rows).col(i);
        double max_conf;
        cv::Point max_loc;
        cv::minMaxLoc(scores, nullptr, &max_conf, nullptr, &max_loc);
        conf_idx_pairs.push_back({ max_conf, i });
    }

    std::sort(conf_idx_pairs.begin(), conf_idx_pairs.end(), std::greater<std::pair<float, int>>());

    // NMS - 객체 탐지(Object Detection) 모델에서 중복된 바운딩 박스를 제거하는 알고리즘
    for (size_t i = 0; i < conf_idx_pairs.size(); ++i) {
        int idx1 = conf_idx_pairs[i].second;
        if (suppressed[idx1]) continue;

        cv::Mat scores = output.rowRange(4, output.rows).col(idx1);
        double max_conf;
        cv::Point max_loc;
        cv::minMaxLoc(scores, nullptr, &max_conf, nullptr, &max_loc);
        int class_id = max_loc.y;

        if (!valid_class_checker_(class_id, class_names_[class_id]) || max_conf < min_conf_) continue;

        ObjectBBox bbox1(
            class_names_[class_id],
            class_id,
            max_conf,
            output.at<float>(0, idx1),
            output.at<float>(1, idx1),
            output.at<float>(2, idx1),
            output.at<float>(3, idx1),
            scale.x,
            scale.y);
        valid_boxes.push_back(bbox1);

        for (size_t j = i + 1; j < conf_idx_pairs.size(); ++j) {
            int idx2 = conf_idx_pairs[j].second;
            if (suppressed[idx2]) continue;

            ObjectBBox bbox2(
                class_names_[class_id],
                class_id,
                output.at<float>(4 + class_id, idx2),
                output.at<float>(0, idx2),
                output.at<float>(1, idx2),
                output.at<float>(2, idx2),
                output.at<float>(3, idx2),
                scale.x,
                scale.y);

            if (calculateIoU(bbox1, bbox2) > iou_thresh_) {
                suppressed[idx2] = true;
            }
        }
    }

    return valid_boxes;
}

std::vector<ObjectBBox> Yolo11::detect(const cv::Mat& image) {
    assert(!image.empty() && image.type() == CV_8UC3 && "Invalid input image");
    cv::Size original_size = image.size();

    cv::Mat blob = preprocess(image);

    net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    assert(outputs.size() == 1 && "Unexpected number of outputs");
    // N x (4 + C)
    // N: 탐지된 바운딩 박스 수
    // 4: 바운딩 박스 좌표(예 : [x, y, w, h] 또는[x1, y1, x2, y2])
    // C : 클래스 확률(클래스 개수에 해당)
    cv::Mat rawOutput = outputs[0].reshape(0, 4 + class_names_.size());

    return postprocess(rawOutput, original_size);
}

std::map<int, std::string> Yolo11::getClassIdNamePairs() const {
    return class_names_;
}

































