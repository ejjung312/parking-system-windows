#include "common_functions.h"

std::string GetResourcePath(const std::string& filename) {
    char path[MAX_PATH] = { 0 };
    GetModuleFileNameA(NULL, path, MAX_PATH);

    std::string dirPath(path);
    size_t pos = dirPath.find_last_of("\\/");
    dirPath = (pos != std::string::npos) ? dirPath.substr(0, pos) : dirPath;

    return dirPath + "\\" + filename;
}