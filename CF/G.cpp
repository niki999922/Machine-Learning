#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <cmath>

using namespace std;

int m, k, h, n, tmp1;
int freeId = 1;

bool is_gini = false;
vector<pair<vector<int>, int>> teach;


struct Node {
    int id = 0;
    int hightOst = 0;

    int featureNumber = 0;
    int b = -100000;
    int classAns = -1;

    bool is_c = false;

    Node *leftChild = nullptr;
    Node *rightChild = nullptr;


    Node(const vector<int> &indexes, int hightOstH) {
        id = freeId;
        hightOst = hightOstH;
        ++freeId;
        buildNode(indexes);
    }

    static unordered_map<int, int> initClassesMap() {
        unordered_map<int, int> maps;
        for (size_t i = 0; i < k; i++) {
            maps[i] = 0;
        }
        return maps;
    }

    static int findMaxClassByChastota(const vector<int> &indexes) {
        auto tmp = initClassesMap();
        for (auto &el: indexes) {
            tmp[teach[el].second] += 1;
        }
        int bestKey = 0;
        for (auto &p : tmp) {
            if (p.second > tmp[bestKey]) {
                bestKey = p.first;
            }
        }
        return bestKey;
    }

    void buildNode(const vector<int> &indexes) {
        if (hightOst == 1) {
            classAns = findMaxClassByChastota(indexes);
            return;
        }
        int bestFeature = 0;
        int bestB = 100000;
        double bestScore = 100000;

        auto all_indexes = initClassesMap();

        for (auto &el: indexes) {
            all_indexes[teach[el].second] += 1;
        }

        int left_best_ind = 0;

        for (size_t featureI = 0; featureI < m; featureI++) {
            vector<pair<int, int>> hello;
            hello.reserve(indexes.size());

            for (auto &el: indexes) {
                hello.emplace_back(teach[el].first[featureI], el);
            }
            sort(hello.begin(), hello.end());
            int left_ind = 0;

            auto leftClassesMap = initClassesMap();
            int left_classes_count_con = 0;
            int right_classes_count_con = hello.size();
            double score;

            while (left_ind < hello.size()) {
                int znash = hello[left_ind].first;
                while (left_ind < hello.size() && znash == hello[left_ind].first) {
                    left_classes_count_con += 1;
                    right_classes_count_con -= 1;
                    leftClassesMap[teach[hello[left_ind].second].second] += 1;
                    left_ind += 1;
                }

                if (is_gini) {
                    score = evalGini(leftClassesMap, left_classes_count_con, all_indexes, true) * left_classes_count_con +
                            evalGini(leftClassesMap, right_classes_count_con, all_indexes, false) * right_classes_count_con;
                } else {
                    score = evalEntrop(leftClassesMap, left_classes_count_con, all_indexes, true) *
                            left_classes_count_con +
                            evalEntrop(leftClassesMap, right_classes_count_con, all_indexes, false) *
                            right_classes_count_con;
                }
                if (bestScore > score) {
                    bestScore = score;
                    left_best_ind = left_ind;
                    bestB = hello[left_ind].first;
                    bestFeature = featureI;
                }
            }
        }

        vector<pair<int, int>> hello;
        hello.reserve(indexes.size());
        for (auto &el: indexes) {
            hello.emplace_back(teach[el].first[bestFeature], el);
        }
        sort(hello.begin(), hello.end());

        vector<int> leftListBEst, rightListBEst;
        for (size_t i = 0; i < hello.size(); i++) {
            if (i < left_best_ind) {
                leftListBEst.push_back(hello[i].second);
            } else {
                rightListBEst.push_back(hello[i].second);
            }
        }

        if (leftListBEst.empty() || rightListBEst.empty()) {
            classAns = findMaxClassByChastota(indexes);
            is_c = true;
            return;
        }
        leftChild = new Node(leftListBEst, hightOst - 1);
        rightChild = new Node(rightListBEst, hightOst - 1);
        b = bestB;
        featureNumber = bestFeature;
    }


    static double
    evalEntrop(unordered_map<int, int> &childs, int elemsSum, unordered_map<int, int> &all_indexes, bool is_left) {
        double res = 0;
        double p;
        int x;
        for (auto &el: childs) {
            if (is_left) {
                x = el.second;
            } else {
                x = all_indexes[el.first] - el.second;
            }
            if (x != 0) {
                p = static_cast<double>(x) / elemsSum;
                res -= (p * log(p));
            }
        }
        return res;
    }

    static double
    evalGini(unordered_map<int, int> &childs, int elemsSum, unordered_map<int, int> &all_indexes, bool is_left) {
        double res = 0;
        double p;
        int x;
        for (auto &el: childs) {
            if (is_left) {
                x = el.second;
            } else {
                x = all_indexes[el.first] - el.second;
            }
            if (x != 0) {
                p = static_cast<double>(x) / elemsSum;
                res += (p * p);
            }
        }
        return 1 - res;
    }

    void printNode(vector<pair<int, string>>& listRes) {
        if (hightOst == 1 || is_c) {
            listRes.emplace_back(id, "C " + to_string(classAns + 1));
        } else {
            listRes.emplace_back(id, "Q " + to_string(featureNumber + 1) + " " + to_string(b) + " " + to_string(leftChild->id)+ " " + to_string(rightChild->id));
            leftChild->printNode(listRes);
            rightChild->printNode(listRes);
        }
    }
};

struct Tree {
    Node *head = nullptr;

    Tree() {
        vector<int> indexes;
        for(size_t i = 0; i < teach.size(); i++) {
            indexes.push_back(i);
        }
        head = new Node(indexes, h + 1);
    }


    void printTree() const {
        vector<pair<int, string>> results;
        head->printNode(results);
        sort(results.begin(),results.end());

        cout << (freeId - 1) << endl;
        for(auto& el: results) {
            cout << el.second << endl;
        }
    }
};


int main() {
    cin >> m >> k >> h >> n;
    teach.reserve(n);
    for (size_t i = 0; i < n; i++) {
        vector<int> features;
        features.reserve(m);
        for (size_t j = 0; j < m; j++) {
            cin >> tmp1;
            features.emplace_back(tmp1);
        }
        cin >> tmp1;
        teach.emplace_back(features, tmp1 - 1);
    }

    if (teach.size() > 50) {
        is_gini = true;
    }
    Tree tree_ML = Tree();
    tree_ML.printTree();

    return 0;
}