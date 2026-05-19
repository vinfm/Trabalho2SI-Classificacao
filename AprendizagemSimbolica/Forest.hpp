#ifndef FOREST_HPP
#define FOREST_HPP
#include "DecisionTree.hpp"
#include <vector>

class Forest {
public:
    std::vector<DecisionTree*> trees;
    Forest(int n_trees) : n_trees(n_trees) {}
    ~Forest() {
        for (DecisionTree* tree : trees) {
            delete tree;
        }
    }
    void addTree(DecisionTree* tree) {
        trees.push_back(tree);
    }
    int predictClassification(const std::vector<double> &features);
    double predictRegression(const std::vector<double> &features);

private:
    int n_trees;
};

#endif