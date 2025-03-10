#include "centroidtracker.h"

using namespace std;

CentroidTracker::CentroidTracker(int maxDisappeared) {
    this->nextObjectID = 0;
    this->maxDisappeared = maxDisappeared;
}

double CentroidTracker::calcDistance(double x1, double y1, double x2, double y2) {
    double x = x1 - x2;
    double y = y1 - y2;
    double dist = sqrt((x * x) + (y * y));       // ��Ŭ����� �Ÿ� ���

    return dist;
}

void CentroidTracker::register_Object(int cX, int cY) {
    int object_ID = this->nextObjectID;
    this->objects.push_back({ object_ID, {cX, cY} });
    this->disappeared.insert({ object_ID, 0 });
    this->nextObjectID += 1;
}

/*
* ���Ϳ��� pos ������ �ּҰ� ��ġ �˻�
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
* boxes: �ڵ����� ������ ��ü�� ����
*/
std::vector<std::pair<int, std::pair<int, int>>> CentroidTracker::update(std::vector<ObjectBBox> boxes) {
    if (boxes.empty()) { // ���� ����� ���� ���

        // ���� ���� ��ü�� �� ���� ������ �ʴ� ��ü ����
        auto it = this->disappeared.begin();
        while (it != this->disappeared.end()) {
            it->second++; // ����� ������ �� ����
            if (it->second > this->maxDisappeared) { // ��ü�� maxDisappeared �����ӵ��� ������ ������ ���� ��󿡼� ����
                // remove_if: ������ ������ �����ϴ� ��ҵ��� ������ ������ �̵��ϰ�,
                //              �������� ���� ��ҵ��� ������ ������ ä �պκ��� ä��
                // erase: ���Ϳ��� ����
                /*this->objects.erase(remove_if(this->objects.begin(), this->objects.end(), 
                    [it](auto& elem) {
                        return elem.first == it->first;
                    }
                ), this->objects.end());*/

                // ������ ��Ҹ� �����̳� �������� �ű�
                auto it_ = remove_if(this->objects.begin(), this->objects.end(), 
                    [it](auto& elem) {
                        return elem.first == it->first;
                    }
                );
                // ȹ���� it_�� ���� ��� ����
                this->objects.erase(it_, this->objects.end());

                // ��ü�� �̵� ��� ����
                this->path_keeper.erase(it->first);

                // ����Ʈ���� �ش� ��ü ����
                it = this->disappeared.erase(it);
            }
            else {
                ++it;
            }
        }
        return this->objects;
    }

    // �ٿ���ڽ� �߽��� ����Ʈ
    vector<pair<int, int>> inputCentroids;
    for (auto b : boxes) {
        int cX = b.cx;
        int cY = b.cy;
        // make_pair: std::pair<int, int> ����
        inputCentroids.push_back(make_pair(cX, cY));
    }

    if (this->objects.empty()) {
        // Ʈ��ŷ���� ��ü�� ���ٸ� �ٿ���ڽ� �߽��� ����Ʈ �߰�
        for (auto& i : inputCentroids) {
            // i.first: pair<int, int>�� ù ��° ����
            // i.second: pair<int, int>�� �� ��° ����
            this->register_Object(i.first, i.second);
        }
    } else {
        // Ʈ��ŷ���� ��ü�� �ִٸ�
        // ���� ��ü�� �ű� ��ü ��Ī
        vector<int> objectIDs;
        vector<pair<int, int>> objectCentroids;
        for (auto object : this->objects) {
            objectIDs.push_back(object.first);
            // object.second.first: pair�� pair�� ù��° int
            // object.second.second: pair�� pair�� �ι�° int
            objectCentroids.push_back(make_pair(object.second.first, object.second.second));
        }

        // �Ÿ� ���
        // Distances: 2D �Ÿ� ��ķ� �� ���� ��ü(��)�� ���ο� ��ü(��) ���� �Ÿ� ���� ����
        vector<vector<float>> Distances;
        for (int i = 0; i < objectCentroids.size(); ++i) {
            vector<float> temp_D;
            // vector(inputCentroids)�� size()�Լ��� size_type ��ȯ
            // std::size_t(��ȣ ���� ����)�� ���ǵǾ� �־� ���� ���� ���� ũ�� ǥ���� ����
            for (vector<vector<int>>::size_type j = 0; j < inputCentroids.size(); ++j) {
                // ��Ŭ����� �Ÿ� ���
                double dist = calcDistance(objectCentroids[i].first, objectCentroids[i].second, inputCentroids[j].first, inputCentroids[j].second);

                temp_D.push_back(dist);
            }
            Distances.push_back(temp_D);
        }

        vector<int> cols; // Distances[i](i ��° ��)���� ���� ����� ��ü�� ��(col) �ε���
        vector<int> rows;

        // �ּڰ��� ���� �ε��� ����
        for (auto v : Distances) {
            auto temp = findMin(v); // �� �� v���� �ּڰ��� ���� ��(col) �ε��� ��ȯ
            cols.push_back(temp);
        }

        // �������� ���� �� D_Copy�� ����
        vector<vector<float>> D_copy;
        for (auto v : Distances) {
            sort(v.begin(), v.end());
            D_copy.push_back(v);
        }

        vector<pair<float, int>> temp_rows;
        int k = 0;
        for (auto i : D_copy) {
            // i[0]: vector<vector<float>> �� ù��° ����� float ��
            temp_rows.push_back(make_pair(i[0], k)); // (�ּ� �Ÿ� ��, �ش� ���� �ε���) ����
            k++;
        }
        for (auto const& x : temp_rows) {
            rows.push_back(x.second);
        }

        set<int> usedRows;
        set<int> usedCols;

        for (int i = 0; i < rows.size(); i++) {
            // �̹� ó���� �� �Ǵ� ���̸� �ǳʶ�
            if (usedRows.count(rows[i]) || usedCols.count(cols[i])) { continue; }
            
            // ���� row �ε����� �ش��ϴ� object ID ��������
            int objectID = objectIDs[rows[i]];

            // objects �迭���� �ش� object ID�� ã��
            for (int t = 0; t < this->objects.size(); t++) {
                if (this->objects[t].first == objectID) {
                    // ���ο� �߽������� ������Ʈ
                    this->objects[t].second.first = inputCentroids[cols[i]].first; // cx
                    this->objects[t].second.second = inputCentroids[cols[i]].second; // cy
                }
            }
            // ��ü ����� ���� �ʱ�ȭ
            this->disappeared[objectID] = 0;

            // ����� ��� �� �߰�
            usedRows.insert(rows[i]);
            usedCols.insert(cols[i]);
        }

        // ��Ī�� �ε����� ������ ������ ��ü���� �ε��� ���
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

        // ���� ��Ī���� ����(�˻���� ����)��� ���� ã��
        // objRows - usedRows ������ �����Ͽ� objRows ���������� usedRows���� ���� ��ҵ��� unusedRows�� ����
        set_difference(objRows.begin(), objRows.end(), usedRows.begin(), usedRows.end(), inserter(unusedRows, unusedRows.begin()));
        set_difference(inpCols.begin(), inpCols.end(), usedCols.begin(), usedCols.end(), inserter(unusedCols, unusedCols.begin()));


        // ������ü >= �ű԰�ü ==> �Ϻ� ��ü�� ������� ���ɼ� ����
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
            // ������ü < �ű԰�ü ==> ���� ��ü�� ��Ī���� ���� ���ο� ��ü ���ɼ�
            for (auto col : unusedCols) {
                this->register_Object(inputCentroids[col].first, inputCentroids[col].second);
            }
        }
    }

    if (!objects.empty()) {
        // ��ü�� �̵� ��� ����
        for (auto obj : objects) {
            if (path_keeper[obj.first].size() > 30) {
                path_keeper[obj.first].erase(path_keeper[obj.first].begin());
            }
            path_keeper[obj.first].push_back(make_pair(obj.second.first, obj.second.second)); // cx, cy
        }
    }

    return this->objects;
}