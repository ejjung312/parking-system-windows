#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <set>
#include <algorithm>
#include <iterator>

#include <opencv2/dnn.hpp>

#include "objectbbox.h"

class CentroidTracker {
public:
    // 묵시적 형변환 막기
    explicit CentroidTracker(int maxDisappeared);

    void register_Object(int cX, int cY);

    std::vector<std::pair<int, std::pair<int, int>>> update(std::vector<ObjectBBox> boxes);

    // <ID, centroids>
    std::vector<std::pair<int, std::pair<int, int>>> objects;

    //make buffer for path tracking
    std::map<int, std::vector<std::pair<int, int>>> path_keeper;
private:
    int maxDisappeared;

    int nextObjectID;

    static double calcDistance(double x1, double y1, double x2, double y2);

    // <ID, count>
    std::map<int, int> disappeared;
};