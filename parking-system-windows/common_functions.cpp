#include "common_functions.h"
#include "objectbbox.h"

#include <opencv2/core/ocl.hpp>

std::string GetResourcePath(const std::string& filename) {
    // MAX_PATH: Windows���� ���ǵ� �ִ� ��� ����
    // {0}���� �ʱ�ȭ�Ͽ� �迭�� ��� ��Ҹ� \0(Null ����)�� ä��
    char path[MAX_PATH] = { 0 };
    // ���� ���� ���� ���� ������ ��ü ��θ� �������� Windows API �Լ�
    // NULL: ���� �������� ���(���� ����)�� ���
    // path: path�� ����
    GetModuleFileNameA(NULL, path, MAX_PATH);

    // std::string dirPath = "path ��";
    std::string dirPath(path);
    // ���� ���� ��ο��� ������ ���͸� ������(\\ �Ǵ� /)�� ��ġ ��ȯ
    size_t pos = dirPath.find_last_of("\\/");
    // ��ġ���� ã���� ��� pos ��ġ���� �ڸ� �κ� ��ȯ
    dirPath = (pos != std::string::npos) ? dirPath.substr(0, pos) : dirPath;

    return dirPath + "\\" + filename;
}

/**
 * IoU - �� ���� ����(������ ������ ���� ����)�� ��ġ�� ������ ��Ÿ����, ���� Ŭ���� ��Ȯ���� ����
 * IoU = ��ģ����(intersection) / ��ü����(union)
 */
float calculateIoU(const ObjectBBox& box1, const ObjectBBox& box2) {
    auto intersection = box1.rect & box2.rect;
    if (intersection.empty()) return 0.0f;

    float intersection_area = intersection.area();
    float union_area = box1.rect.area() + box2.rect.area() - intersection_area;

    return intersection_area / union_area;
}

///////////////////////////////////////////////////////////////////

void selectDNNBackendAndTarget(cv::dnn::Net& net) {
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        std::cout << "Using CUDA backend.\n";
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else if (cv::ocl::haveOpenCL()) {
        std::cout << "Using OpenCL backend.\n";
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
    }
    else {
        std::cout << "Using CPU backend.\n";
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}