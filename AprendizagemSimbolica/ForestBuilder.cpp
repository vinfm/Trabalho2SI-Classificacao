#include "ForestBuilder.hpp"
#include <random>
#include <algorithm>
#include <cmath>

ForestBuilder::ForestBuilder(TreeType treeType, Dataset *trainData, Dataset *subtrainData,
     int min_samples_leaf, int max_depth, int min_samples_split, double min_impurity_split,
      std::size_t n_features, int n_trees)
{
    treeType_ = treeType;
    trainData_ = trainData;
    subtrainData_ = subtrainData;
    min_samples_leaf_ = min_samples_leaf;
    max_depth_ = max_depth;
    min_samples_split_ = min_samples_split;
    min_impurity_split_ = min_impurity_split;
    n_features_ = n_features;
    n_trees_ = n_trees;
    forest_ = nullptr;

    max_features_ = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(n_features_)))));
}

Forest *ForestBuilder::buildForest()
{
    forest_ = new Forest(n_trees_);

    for (int i = 0; i < n_trees_; ++i) {
        Dataset* bootstrap_data = BootStrapSample(trainData_, i);
        TreeBuilder* builder = new TreeBuilder
        (treeType_, bootstrap_data, subtrainData_, min_samples_leaf_, max_depth_,
             min_samples_split_, min_impurity_split_, n_features_);
        // Random subspace: cada split avalia apenas max_features_ atributos
        builder->setMaxFeatures(max_features_);
        builder->setRandomSeed(static_cast<unsigned>(i) + 1u);
        DecisionTree* tree = builder->CARTBuild();
        forest_->addTree(tree);
        delete bootstrap_data;
        delete builder;
    }

    return forest_;
}

Dataset *ForestBuilder::BootStrapSample(const Dataset *data, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, static_cast<int>(data->nrows()) - 1);
    
    Dataset* bootstrap_data = new Dataset;
    bootstrap_data->X.resize(data->nrows());
    bootstrap_data->Y.resize(data->nrows());
    
    for (std::size_t i = 0; i < data->nrows(); ++i) {
        int random_idx = dis(gen);
        bootstrap_data->X[i] = data->X[random_idx];
        bootstrap_data->Y[i] = data->Y[random_idx];
    }
    
    return bootstrap_data;
}
