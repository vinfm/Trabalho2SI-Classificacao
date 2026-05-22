#include "Forest.hpp"
#include <algorithm>
#include <numeric>
int Forest::predictClassification(const std::vector<double> &features)
{
    std::vector<int> predictions;
    for (DecisionTree* tree : trees) {
        int prediction = tree->predictClassification(features);
        if (prediction != -1) {
            predictions.push_back(prediction);
        }
    }
    
    // calculate majority vote
    if (predictions.empty()) {
        return -1; // No predictions available
    }
    std::vector<int> classCounts(*std::max_element(predictions.begin(), predictions.end()) + 1, 0);
    for (int pred : predictions) {
        classCounts[pred]++;
    }
    return std::distance(classCounts.begin(), std::max_element(classCounts.begin(), classCounts.end()));
}

// Filtro de rgressão prediction != 0 removido
double Forest::predictRegression(const std::vector<double> &features)
{
    if (trees.empty()) return 0.0;
    double sum = 0.0;
    for (DecisionTree* tree : trees) {
        sum += tree->predictRegression(features);
    }
    return sum / static_cast<double>(trees.size());
}
