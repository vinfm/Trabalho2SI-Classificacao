#ifndef NODE_HPP
#define NODE_HPP

#include <cstddef>
#include <vector>
class Node {
public:
    bool isLeaf;
    std::size_t featureIndex;
    double threshold;        
    std::vector<double> outputsRegression; // For regression trees
    std::vector<int> outputsClassification; // For classification trees
    std::size_t left; 
    std::size_t right;
    double impurity;
    int classificationOutput();
    double regressionOutput();
};

#endif