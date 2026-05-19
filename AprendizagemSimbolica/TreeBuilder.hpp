#ifndef TREEBUILDER_HPP
#define TREEBUILDER_HPP
#include "DecisionTree.hpp"
#include <vector>
#include "Splitter.hpp"
#include "Cart.hpp"
#include <cmath>
#include <random>

class TreeBuilder {
private:
    TreeType treeType_;
    Dataset* trainData_; // Dataset for training
    Dataset* subtrainData_; // Subset of data for current node
    Splitter splitter_;
    int min_samples_leaf_;
    int max_depth_;
    int min_samples_split_;
    double min_impurity_split_;
    std::size_t n_features_;
    CART* tree_;
    CriterionType impurity_measure_;
    std::size_t max_features_;
    std::mt19937 gen_;
public:
    TreeBuilder(TreeType treeType, Dataset* trainData, Dataset* subtrainData, int min_samples_leaf, int max_depth, int min_samples_split, double min_impurity_split, std::size_t n_features);
    DecisionTree* CARTBuild();
    void setMaxFeatures(std::size_t max_features) {
        max_features_ = max_features;
    }
    void setRandomSeed(unsigned seed) {
        gen_.seed(seed);
    }
    void CalculateNodeOutput(Node &node);
private:

};
#endif // TREEBUILDER_HPP