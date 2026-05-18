#include "DecisionTree.hpp"

DecisionTree::DecisionTree(Node root)
{
    nodes.push_back(root);
    rootIndex = 0;
}

DecisionTree::~DecisionTree()
{
}
