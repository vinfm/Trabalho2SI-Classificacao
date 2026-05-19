#include "DecisionTree.hpp"

DecisionTree::DecisionTree(Node root, TreeType type) : type(type)
{
    nodes.push_back(root);
    rootIndex = 0;
}

DecisionTree::~DecisionTree()
{
}
