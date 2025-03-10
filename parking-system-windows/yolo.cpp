#include "yolo.h"
#include "debug.h"
#include "common_functions.h"

#include <fstream>

void Yolo11::loadClassNames(const std::string& names_file) {
    std::ifstream file(names_file);
    std::cout << "file name: " << names_file << std::endl;
    assert(file.is_open() && ("Failed to open class names file " + names_file).c_str());

    std::string line;
    int class_id = 0;

    while (std::getline(file, line)) {
        // trim
        /*
        ' ' (�����̽�)
        '\t' (��)
        '\n' (����)
        '\r' (ĳ���� ����)
        '\f' (�� �ǵ�)
        '\v' (���� ��)
        */
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v")); // ������ �ƴ� ù��° ���ڿ� ��ġ�� ã�� ã�� �ε����� �߶󳻾� ���� ����
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1); // ������ �ƴ� ������ ������ ��ġ�� ã�� ������ ������ ���� ��ġ(+1)���� ���ڿ� ������ ����

        if (!line.empty()) {
            class_names_[class_id++] = line;
        }
    }

    assert(!class_names_.empty() && "No class names loaded from file `coco.names`");
    DEBUG_PRINT("Loaded" << class_names_.size() << " class names");
}

Yolo11::Yolo11(const std::string& model_path, float min_conf, float iou_thresh, ClassChecker valid_class_checker, const std::string& names_file) 
    : min_conf_(min_conf), iou_thresh_(iou_thresh) {
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

    // Ŭ���� ����
    std::string resolved_names_file = names_file;
    if (resolved_names_file.empty()) {
        size_t last_slash_idx = model_path.find_last_of("\\/");
        // std::string::npos - -1 ���� ������ ����� ���ڿ��� ���� �� ��ȯ
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

    // ���׿�����
    valid_class_checker_ = valid_class_checker ? valid_class_checker : [](int, const std::string&) { return true; };
}

cv::Mat Yolo11::preprocess(const cv::Mat& image) {
    cv::Mat resized;
    cv::resize(image, resized, input_size_);
    DEBUG_PRINT_MAT_SHAPE(resized);

    cv::Mat blob;
    // blobFromImage: ������ ��(cn::dnn::Net)�� ó���� �� �ִ� ���·� ��ȯ
    //                (batch_size, channels, height, width) �������� ��ȯ.(NCHW)
    // �Է��̹���, ���, ����ȭ, �Է�ũ��, ��հ�(�Է� �̹������� �� ��հ�), true(BGR->RGB), false(�Է��̹����� input_size_ ũ��� ������ �����ϸ鼭 �е��߰�), float32�� ��ȯ
    cv::dnn::blobFromImage(resized, blob, 1.0/255.0, input_size_, cv::Scalar(), true, false, CV_32F);
    DEBUG_PRINT_MAT_SHAPE(blob);

    return blob;
}

std::vector<ObjectBBox> Yolo11::postprocess(const cv::Mat& output, const cv::Size& original_size) {
    DEBUG_PRINT_MAT_SHAPE(output);
    assert(output.dims == 2 && output.cols > 0 &&
        output.rows == (4 + class_names_.size()) &&
        "Invalid output shape");

    std::vector<ObjectBBox> valid_boxes;
    // suppressed������ output.cols ũ�⸸ŭ false�� ���� �ʱ�ȭ
    std::vector<bool> suppressed(output.cols, false);
    // cv::Point2f - 2D ��ǥ�� �����ϴ� ����ü. (x,y) ���� float Ÿ������ ����
    // ���� ũ��� �Է� ũ���� ������ ����Ͽ� ����
    cv::Point2f scale(
        static_cast<float>(original_size.width) / input_size_.width,
        static_cast<float>(original_size.height) / input_size_.height
    );

    // �ִ� confidence ã��
    std::vector<std::pair<float, int>> conf_idx_pairs; // <�ִ� confidence score, ��ü �ε���>
    for (int i = 0; i < output.cols; ++i) {
        // 0~3: �ٿ�� �ڽ� ��ǥ(x, y, width, height)
        // 4~N: �� Ŭ������ Ȯ��(scores)
        // ���� ���� conf�� ���� Ŭ���� ã��
        cv::Mat scores = output.rowRange(4, output.rows).col(i);
        double max_conf;
        cv::Point max_loc;
        // scores ��Ŀ��� �ִ밪(max_conf)�� �ش� ��ġ(max_loc.y)�� ã��
        cv::minMaxLoc(scores, nullptr, &max_conf, nullptr, &max_loc);

        conf_idx_pairs.push_back({ max_conf, i });
    }

    // conf�� ���� ��ü���� �������� ����
    std::sort(conf_idx_pairs.begin(), conf_idx_pairs.end(), std::greater<std::pair<float, int>>());

    // NMS - ��ü Ž��(Object Detection) �𵨿��� �ߺ��� �ٿ�� �ڽ��� �����ϴ� �˰���
    for (size_t i = 0; i < conf_idx_pairs.size(); ++i) {
        int idx1 = conf_idx_pairs[i].second;
        if (suppressed[idx1]) continue;

        // ���� ���� conf�� ���� Ŭ���� ã��
        cv::Mat scores = output.rowRange(4, output.rows).col(idx1); // idx1��° ��ü�� Ŭ���� Ȯ�� ��
        double max_conf;
        cv::Point max_loc;
        cv::minMaxLoc(scores, nullptr, &max_conf, nullptr, &max_loc);
        int class_id = max_loc.y;

        // ��ȿ�� Ŭ������ �ƴϰų� �ŷڵ��� ������ ���� ��ü��
        if (!valid_class_checker_(class_id, class_names_[class_id]) || max_conf < min_conf_) continue;

        ObjectBBox bbox1(
            class_names_[class_id],
            class_id,
            max_conf,
            output.at<float>(0, idx1), // cx
            output.at<float>(1, idx1), // cy
            output.at<float>(2, idx1), // w
            output.at<float>(3, idx1), // h
            scale.x,
            scale.y);
        valid_boxes.push_back(bbox1);

        // NMS ����
        for (size_t j = i + 1; j < conf_idx_pairs.size(); ++j) {
            int idx2 = conf_idx_pairs[j].second;
            if (suppressed[idx2]) continue;

            ObjectBBox bbox2(
                class_names_[class_id],
                class_id,
                output.at<float>(4 + class_id, idx2),
                output.at<float>(0, idx2), // cx
                output.at<float>(1, idx2), // cy
                output.at<float>(2, idx2), // w
                output.at<float>(3, idx2), // h
                scale.x,
                scale.y);

            // �� �ڽ��� IoU(��ġ�� ����)�� ������ �Ӱ谪���� ũ�ٸ� �ߺ��� ������ �����ϰ� ����
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
    // net_.getUnconnectedOutLayersNames() - ��� ���̾�(������ ���̾�)�� �̸� ��ȯ
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    assert(outputs.size() == 1 && "Unexpected number of outputs");
    // N x (4 + C)
    // N: Ž���� �ٿ�� �ڽ� ��
    // 4: �ٿ�� �ڽ� ��ǥ(�� : [x, y, w, h] �Ǵ�[x1, y1, x2, y2])
    // C: Ŭ���� Ȯ��(Ŭ���� ������ �ش�)
    // 1���� ��ü�� (4 + Ŭ���� ����) ���� ���� ������ ��� ����
    cv::Mat rawOutput = outputs[0].reshape(0, 4 + class_names_.size());

    return postprocess(rawOutput, original_size);
}

std::map<int, std::string> Yolo11::getClassIdNamePairs() const {
    return class_names_;
}

































