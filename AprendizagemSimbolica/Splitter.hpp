#ifndef SPLITTER_HPP
#define SPLITTER_HPP
#include <vector>
#include "Criteria.hpp"
#include <random>
class Split {
public:
    int featureIndex;
    double threshold;
    double impurity; // weighted impurity after split
    double gain;     // impurity reduction from parent to children
    int postSplit;
    std::vector<std::size_t> data_left;
    std::vector<std::size_t> data_right;
    Split();
};

class Splitter {
public:
    std::vector<std::vector<double>> data_;
    std::vector<std::vector<double>> output_data_;
    std::vector<std::pair<int, double>> sample_map_;
    int min_samples_leaf_;
    std::size_t max_features_;
    std::mt19937 *gen_;
    std::vector<int> feature_order_;
    bool shuffle_features_;
    std::size_t n_samples_total_;
    std::size_t n_features_;
    std::vector<std::size_t> samples_split_;
    Splitter(CriterionType criterionType);
    Splitter(CriterionType criterionType,
             const std::vector<std::vector<double>> &data,
             const std::vector<std::vector<double>> &output_data,
             std::size_t n_features,
             std::size_t max_features,
             int min_samples_leaf);
    void setRandomGenerator(std::mt19937* gen) {
        gen_ = gen;
    }
    void SplitNode(Split &split);
private:
    Criteria criteria_;
    CriterionType criterionType_;
};
#endif