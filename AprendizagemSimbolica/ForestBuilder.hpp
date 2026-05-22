#ifndef FOREST_BUILDER_HPP
#define FOREST_BUILDER_HPP
#include "Forest.hpp"
#include "TreeBuilder.hpp"
class ForestBuilder {
private:
    TreeType treeType_;
    Dataset* trainData_;
    Dataset* subtrainData_;
    int min_samples_leaf_;
    int max_depth_;
    int min_samples_split_;
    double min_impurity_split_;
    std::size_t n_features_;
    std::size_t max_features_;
    std::vector<Dataset*> subtrainDatasets_;
    Forest* forest_;
    int n_trees_;
public:
    ForestBuilder(TreeType treeType, Dataset* trainData, Dataset* subtrainData, int min_samples_leaf, int max_depth, int min_samples_split, double min_impurity_split, std::size_t n_features, int n_trees);
    Forest* buildForest();
    Dataset* BootStrapSample(const Dataset* data, unsigned seed);
};

#endif