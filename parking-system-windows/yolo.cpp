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
        ' ' (스페이스)
        '\t' (탭)
        '\n' (개행)
        '\r' (캐리지 리턴)
        '\f' (폼 피드)
        '\v' (수직 탭)
        */
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v")); // 공백이 아닌 첫번째 문자열 위치를 찾고 찾은 인덱스로 잘라내어 공백 제거
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1); // 공백이 아닌 마지막 문자의 위치를 찾아 공백이 끝나는 다음 위치(+1)부터 문자열 끝까지 삭제

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

    // 클래스 파일
    std::string resolved_names_file = names_file;
    if (resolved_names_file.empty()) {
        size_t last_slash_idx = model_path.find_last_of("\\/");
        // std::string::npos - -1 값을 가지는 상수로 문자열이 없을 때 반환
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
    // blobFromImage: 딥러닝 모델(cn::dnn::Net)이 처리할 수 있는 형태로 변환
    //                (batch_size, channels, height, width) 형식으로 변환.(NCHW)
    // 입력이미지, 결과, 정규화, 입력크기, 평균값(입력 이미지에서 뺄 평균값), true(BGR->RGB), false(입력이미지를 input_size_ 크기로 비율을 유지하면서 패딩추가), float32로 변환
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
    // suppressed변수에 output.cols 크기만큼 false로 벡터 초기화
    std::vector<bool> suppressed(output.cols, false);
    // cv::Point2f - 2D 좌표를 저장하는 구조체. (x,y) 값을 float 타입으로 저장
    // 원본 크기와 입력 크기의 비율을 계산하여 저장
    cv::Point2f scale(
        static_cast<float>(original_size.width) / input_size_.width,
        static_cast<float>(original_size.height) / input_size_.height
    );

    // 최대 confidence 찾기
    std::vector<std::pair<float, int>> conf_idx_pairs; // <최대 confidence score, 객체 인덱스>
    for (int i = 0; i < output.cols; ++i) {
        // 0~3: 바운딩 박스 좌표(x, y, width, height)
        // 4~N: 각 클래스의 확률(scores)
        // 가장 높은 conf를 가진 클래스 찾기
        cv::Mat scores = output.rowRange(4, output.rows).col(i);
        double max_conf;
        cv::Point max_loc;
        // scores 행렬에서 최대값(max_conf)과 해당 위치(max_loc.y)를 찾음
        cv::minMaxLoc(scores, nullptr, &max_conf, nullptr, &max_loc);

        conf_idx_pairs.push_back({ max_conf, i });
    }

    // conf가 높은 객체부터 내림차순 정렬
    std::sort(conf_idx_pairs.begin(), conf_idx_pairs.end(), std::greater<std::pair<float, int>>());

    // NMS - 객체 탐지(Object Detection) 모델에서 중복된 바운딩 박스를 제거하는 알고리즘
    for (size_t i = 0; i < conf_idx_pairs.size(); ++i) {
        int idx1 = conf_idx_pairs[i].second;
        if (suppressed[idx1]) continue;

        // 가장 높은 conf를 가진 클래스 찾기
        cv::Mat scores = output.rowRange(4, output.rows).col(idx1); // idx1번째 객체의 클래스 확률 값
        double max_conf;
        cv::Point max_loc;
        cv::minMaxLoc(scores, nullptr, &max_conf, nullptr, &max_loc);
        int class_id = max_loc.y;

        // 유효한 클래스가 아니거나 신뢰도가 낮으면 다음 객체로
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

        // NMS 실행
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

            // 두 박스의 IoU(겹치는 비율)가 설정된 임계값보다 크다면 중복된 것으로 간주하고 제거
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
    // net_.getUnconnectedOutLayersNames() - 출력 레이어(마지막 레이어)의 이름 반환
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    assert(outputs.size() == 1 && "Unexpected number of outputs");
    // N x (4 + C)
    // N: 탐지된 바운딩 박스 수
    // 4: 바운딩 박스 좌표(예 : [x, y, w, h] 또는[x1, y1, x2, y2])
    // C: 클래스 확률(클래스 개수에 해당)
    // 1개의 객체당 (4 + 클래수 개수) 개의 값을 갖도록 행렬 변형
    cv::Mat rawOutput = outputs[0].reshape(0, 4 + class_names_.size());

    return postprocess(rawOutput, original_size);
}

std::map<int, std::string> Yolo11::getClassIdNamePairs() const {
    return class_names_;
}

































