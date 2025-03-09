#include "common_functions.h"
#include "objectbbox.h"

std::string GetResourcePath(const std::string& filename) {
    // MAX_PATH: Windows에서 정의된 최대 경로 길이
    // {0}으로 초기화하여 배열의 모든 요소를 \0(Null 문자)로 채움
    char path[MAX_PATH] = { 0 };
    // 현재 실행 중인 실행 파일의 전체 경로를 가져오는 Windows API 함수
    // NULL: 현재 실행중인 모듈(실행 파일)의 경로
    // path: path에 저장
    GetModuleFileNameA(NULL, path, MAX_PATH);

    // std::string dirPath = "path 값";
    std::string dirPath(path);
    // 실행 파일 경로에서 마지막 디렉터리 구분자(\\ 또는 /)의 위치 반환
    size_t pos = dirPath.find_last_of("\\/");
    // 위치값을 찾았을 경우 pos 위치까지 자른 부분 반환
    dirPath = (pos != std::string::npos) ? dirPath.substr(0, pos) : dirPath;

    return dirPath + "\\" + filename;
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