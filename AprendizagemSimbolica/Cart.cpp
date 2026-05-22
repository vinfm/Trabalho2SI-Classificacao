#include "Cart.hpp"
#include <iostream>
#include "TreeBuilder.hpp"

void CART::saveModel(const std::string &path)
{
}

void CART::loadModel(const std::string &path)
{
}

void CART::printTree()
{
    if (nodes.empty()) {
        std::cout << "Empty tree." << std::endl;
        return;
    }
    std::cout << "CART Decision Tree:" << std::endl;
    if (type == TreeType::Classification) {
        std::cout << "Tree Type: Classification" << std::endl;
    } else {
        std::cout << "Tree Type: Regression" << std::endl;
    }
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const Node &node = nodes[i];
        std::cout << "Node " << i << ": ";
        if (node.isLeaf) {
            std::cout << "Leaf Node, ";
            if (type == TreeType::Classification) {
                std::cout << "Class: " << const_cast<Node&>(node).classificationOutput() << std::endl;
            } else {
                std::cout << "Value: " << const_cast<Node&>(node).regressionOutput() << std::endl;
            }
        } else {
            std::cout << "Feature Index: " << node.featureIndex
                      << ", Threshold: " << node.threshold
                      << ", Impurity: " << node.impurity
                      << ", Left Child: " << node.left
                      << ", Right Child: " << node.right
                      << std::endl;
        }
    }
}
int CART::predictClassification(const std::vector<double> &features)
{
    int idx = 0; // root index
    while (true) {
        if (idx < 0 || idx >= static_cast<int>(nodes.size())) return -1;
        auto &node = nodes[idx];
        if (node.isLeaf) {
            return node.classificationOutput();
        }
        if (node.featureIndex < 0 || node.featureIndex >= static_cast<int>(features.size())) return -1;
        idx = (features[node.featureIndex] <= node.threshold) ? node.left : node.right;
    }
}

double CART::predictRegression(const std::vector<double> &features)
{
    int idx = 0; // root index
    while (true) {
        if (idx < 0 || idx >= static_cast<int>(nodes.size())) return 0.0;
        auto &node = nodes[idx];
        if (node.isLeaf) {
            return node.regressionOutput();
        }
        if (node.featureIndex < 0 || node.featureIndex >= static_cast<int>(features.size())) return 0.0;
        idx = (features[node.featureIndex] <= node.threshold) ? node.left : node.right;
    }
}
