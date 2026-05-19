#ifndef DECISIONTREE_HPP
#define DECISIONTREE_HPP
#include <cstddef>
#include <vector>
#include "Node.hpp"

enum class TreeType { Classification, Regression };

class DecisionTree
{
public:
    std::vector<Node> nodes;
    TreeType type;
    std::size_t rootIndex = 0;
    DecisionTree(Node root, TreeType type): type(type) {
        nodes.push_back(root);
    }
    ~DecisionTree();
    virtual int predictClassification(const std::vector<double> &features) = 0;
    virtual double predictRegression(const std::vector<double>& features) = 0;
    virtual void saveModel(const std::string &path) = 0;
    virtual void loadModel(const std::string &path) = 0;
    virtual void printTree() = 0;
};

#endif