#ifndef NODE_HPP
#define NODE_HPP

#include <cstddef>
#include <vector>
class Node {
public:
    bool isLeaf;
    std::size_t featureIndex;
    double threshold;       
    std::vector<double> valuesRegression; // For regression trees
    std::vector<int> valuesClassification; // For classification trees
    int depth;
    std::size_t left; 
    std::size_t right;
    double impurity;
    std::vector<std::size_t> sampleIndices; // Indices of samples in this node
    std::vector<std::size_t> leftSampleIndices; // Indices of samples in left child
    std::vector<std::size_t> rightSampleIndices; // Indices of samples in right child
    int classificationOutput();
    double regressionOutput();
};

#endif