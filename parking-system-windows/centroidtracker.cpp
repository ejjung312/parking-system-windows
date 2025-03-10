#include "centroidtracker.h"

using namespace std;

CentroidTracker::CentroidTracker(int maxDisappeared) {
    this->nextObjectID = 0;
    this->maxDisappeared = maxDisappeared;
}

double CentroidTracker::calcDistance(double x1, double y1, double x2, double y2) {
    double x = x1 - x2;
    double y = y1 - y2;
    double dist = sqrt((x * x) + (y * y));       // 유클리디안 거리 계산

    return dist;
}

void CentroidTracker::register_Object(int cX, int cY) {
    int object_ID = this->nextObjectID;
    this->objects.push_back({ object_ID, {cX, cY} });
    this->disappeared.insert({ object_ID, 0 });
    this->nextObjectID += 1;
}

/*
* 벡터에서 pos 이후의 최소값 위치 검색
*/
vector<float>::size_type findMin(const vector<float>& v, vector<float>::size_type pos = 0) {
    if (v.size() <= pos) return (v.size());
    vector<float>::size_type min = pos;
    for (vector<float>::size_type i = pos + 1; i < v.size(); i++) {
        if (v[i] < v[min]) min = i;
    }
    return (min);
}

/*
* boxes: 자동차를 감지한 객체의 정보
*/
std::vector<std::pair<int, std::pair<int, int>>> CentroidTracker::update(std::vector<ObjectBBox> boxes) {
    if (boxes.empty()) { // 검출 결과가 없을 경우

        // 추적 중인 객체들 중 현재 보이지 않는 객체 추적
        auto it = this->disappeared.begin();
        while (it != this->disappeared.end()) {
            it->second++; // 사라진 프레임 수 증가
            if (it->second > this->maxDisappeared) { // 객체가 maxDisappeared 프레임동안 보이지 않으면 추적 대상에서 제거
                // remove_if: 삭제할 조건을 만족하는 요소들을 벡터의 끝으로 이동하고,
                //              삭제되지 않은 요소들의 순서를 유지한 채 앞부분을 채움
                // erase: 벡터에서 제거
                /*this->objects.erase(remove_if(this->objects.begin(), this->objects.end(), 
                    [it](auto& elem) {
                        return elem.first == it->first;
                    }
                ), this->objects.end());*/

                // 삭제할 요소를 컨테이너 뒤쪽으로 옮김
                auto it_ = remove_if(this->objects.begin(), this->objects.end(), 
                    [it](auto& elem) {
                        return elem.first == it->first;
                    }
                );
                // 획득한 it_를 통해 요소 삭제
                this->objects.erase(it_, this->objects.end());

                // 객체의 이동 경로 제거
                this->path_keeper.erase(it->first);

                // 리스트에서 해당 객체 제거
                it = this->disappeared.erase(it);
            }
            else {
                ++it;
            }
        }
        return this->objects;
    }

    // 바운딩박스 중심점 리스트
    vector<pair<int, int>> inputCentroids;
    for (auto b : boxes) {
        int cX = b.cx;
        int cY = b.cy;
        // make_pair: std::pair<int, int> 생성
        inputCentroids.push_back(make_pair(cX, cY));
    }

    if (this->objects.empty()) {
        // 트래킹중인 객체가 없다면 바운딩박스 중심점 리스트 추가
        for (auto& i : inputCentroids) {
            // i.first: pair<int, int>의 첫 번째 정수
            // i.second: pair<int, int>의 두 번째 정수
            this->register_Object(i.first, i.second);
        }
    } else {
        // 트래킹중인 객체가 있다면
        // 기존 객체와 신규 객체 매칭
        vector<int> objectIDs;
        vector<pair<int, int>> objectCentroids;
        for (auto object : this->objects) {
            objectIDs.push_back(object.first);
            // object.second.first: pair의 pair의 첫번째 int
            // object.second.second: pair의 pair의 두번째 int
            objectCentroids.push_back(make_pair(object.second.first, object.second.second));
        }

        // 거리 계산
        // Distances: 2D 거리 행렬로 각 기존 객체(행)와 새로운 객체(열) 간의 거리 정보 저장
        vector<vector<float>> Distances;
        for (int i = 0; i < objectCentroids.size(); ++i) {
            vector<float> temp_D;
            // vector(inputCentroids)의 size()함수는 size_type 반환
            // std::size_t(부호 없는 정수)로 정의되어 있어 음수 값이 없는 크기 표현에 적합
            for (vector<vector<int>>::size_type j = 0; j < inputCentroids.size(); ++j) {
                // 유클리디안 거리 계산
                double dist = calcDistance(objectCentroids[i].first, objectCentroids[i].second, inputCentroids[j].first, inputCentroids[j].second);

                temp_D.push_back(dist);
            }
            Distances.push_back(temp_D);
        }

        vector<int> cols; // Distances[i](i 번째 행)에서 가장 가까운 객체의 열(col) 인덱스
        vector<int> rows;

        // 최솟값을 가진 인덱스 저장
        for (auto v : Distances) {
            auto temp = findMin(v); // 각 행 v에서 최솟값을 가진 열(col) 인덱스 반환
            cols.push_back(temp);
        }

        // 오름차순 정렬 후 D_Copy에 저장
        vector<vector<float>> D_copy;
        for (auto v : Distances) {
            sort(v.begin(), v.end());
            D_copy.push_back(v);
        }

        vector<pair<float, int>> temp_rows;
        int k = 0;
        for (auto i : D_copy) {
            // i[0]: vector<vector<float>> 의 첫번째 요소인 float 값
            temp_rows.push_back(make_pair(i[0], k)); // (최소 거리 값, 해당 행의 인덱스) 저장
            k++;
        }
        for (auto const& x : temp_rows) {
            rows.push_back(x.second);
        }

        set<int> usedRows;
        set<int> usedCols;

        for (int i = 0; i < rows.size(); i++) {
            // 이미 처리한 행 또는 열이면 건너뜀
            if (usedRows.count(rows[i]) || usedCols.count(cols[i])) { continue; }
            
            // 현재 row 인덱스에 해당하는 object ID 가져오기
            int objectID = objectIDs[rows[i]];

            // objects 배열에서 해당 object ID를 찾음
            for (int t = 0; t < this->objects.size(); t++) {
                if (this->objects[t].first == objectID) {
                    // 새로운 중심점으로 업데이트
                    this->objects[t].second.first = inputCentroids[cols[i]].first; // cx
                    this->objects[t].second.second = inputCentroids[cols[i]].second; // cy
                }
            }
            // 객체 사라짐 상태 초기화
            this->disappeared[objectID] = 0;

            // 사용한 행과 열 추가
            usedRows.insert(rows[i]);
            usedCols.insert(cols[i]);
        }

        // 매칭된 인덱스를 제외한 나머지 객체들의 인덱스 계산
        set<int> objRows;
        set<int> inpCols;

        for (int i = 0; i < objectCentroids.size(); i++) {
            objRows.insert(i);
        }
        for (int i = 0; i < inputCentroids.size(); i++) {
            inpCols.insert(i);
        }

        set<int> unusedRows;
        set<int> unusedCols;

        // 아직 매칭되지 않은(검사되지 않은)행과 열을 찾음
        // objRows - usedRows 연산을 수행하여 objRows 존재하지만 usedRows에는 없는 요소들을 unusedRows에 저장
        set_difference(objRows.begin(), objRows.end(), usedRows.begin(), usedRows.end(), inserter(unusedRows, unusedRows.begin()));
        set_difference(inpCols.begin(), inpCols.end(), usedCols.begin(), usedCols.end(), inserter(unusedCols, unusedCols.begin()));


        // 기존객체 >= 신규객체 ==> 일부 객체가 사라졌을 가능성 있음
        if (objectCentroids.size() >= inputCentroids.size()) {
            for (auto row : unusedRows) {
                int objectID = objectIDs[row];
                this->disappeared[objectID] += 1;

                if (this->disappeared[objectID] > this->maxDisappeared) {
                    this->objects.erase(remove_if(this->objects.begin(), this->objects.end(), [objectID](auto& elem) {
                        return elem.first == objectID;
                    }), this->objects.end());

                    this->path_keeper.erase(objectID);

                    this->disappeared.erase(objectID);
                }
            }
        } else {
            // 기존객체 < 신규객체 ==> 기존 객체와 매칭되지 않은 새로운 객체 가능성
            for (auto col : unusedCols) {
                this->register_Object(inputCentroids[col].first, inputCentroids[col].second);
            }
        }
    }

    if (!objects.empty()) {
        // 객체의 이동 경로 저장
        for (auto obj : objects) {
            if (path_keeper[obj.first].size() > 30) {
                path_keeper[obj.first].erase(path_keeper[obj.first].begin());
            }
            path_keeper[obj.first].push_back(make_pair(obj.second.first, obj.second.second)); // cx, cy
        }
    }

    return this->objects;
}