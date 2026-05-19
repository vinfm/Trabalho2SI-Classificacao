#ifndef CART_HPP
#define CART_HPP

#include <cstddef>
#include <string>
#include <vector>
#include "Node.hpp"
#include "Dataset.hpp"
#include "DecisionTree.hpp"
class CART : public DecisionTree {
    public:
        CART() = default;
        CART(Node root, TreeType type): DecisionTree(root, type) {}
        ~CART() = default;
        void saveModel(const std::string &path);
        void loadModel(const std::string &path);
        void printTree();
        int predictClassification(const std::vector<double> &features);
        double predictRegression(const std::vector<double> &features);
};
#endif