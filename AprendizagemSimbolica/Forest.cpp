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

double Forest::predictRegression(const std::vector<double> &features)
{
    std::vector<double> predictions;
    for (DecisionTree* tree : trees) {
        double prediction = tree->predictRegression(features);
        if (prediction != 0.0) { // Assuming 0.0 means no prediction, adjust as needed
            predictions.push_back(prediction);
        }
    }
    
    if (predictions.empty()) {
        return 0.0; // No predictions available
    }
    double sum = std::accumulate(predictions.begin(), predictions.end(), 0.0);
    return sum / predictions.size();
}
