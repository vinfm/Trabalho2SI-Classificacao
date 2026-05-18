#ifndef DECISIONTREE_HPP
#define DECISIONTREE_HPP
#include <cstddef>
#include <vector>
#include "Node.hpp"
class DecisionTree
{
public:
    std::vector<Node> nodes;
    TreeType type;
    std::size_t rootIndex = 0;
    DecisionTree(Node root);
    ~DecisionTree();
    virtual int predictClassification(const std::vector<double> &features) = delete;
    virtual double predictRegression(const std::vector<double>& features) = 0;
};

#endif