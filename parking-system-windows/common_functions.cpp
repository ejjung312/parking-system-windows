#include "common_functions.h"
#include "objectbbox.h"

std::string GetResourcePath(const std::string& filename) {
    char path[MAX_PATH] = { 0 };
    GetModuleFileNameA(NULL, path, MAX_PATH);

    std::string dirPath(path);
    size_t pos = dirPath.find_last_of("\\/");
    dirPath = (pos != std::string::npos) ? dirPath.substr(0, pos) : dirPath;

    return dirPath + "\\" + filename;
}

/**
 * IoU - �� ���� ����(������ ������ ���� ����)�� ��ġ�� ������ ��Ÿ����, ���� Ŭ���� ��Ȯ���� ����
 */
float calculateIoU(const ObjectBBox& box1, const ObjectBBox& box2) {
    auto intersection = box1.rect & box2.rect;
    if (intersection.empty()) return 0.0f;

    float intersection_area = intersection.area();
    float union_area = box1.rect.area() + box2.rect.area() - intersection_area;

    return intersection_area / union_area;
}