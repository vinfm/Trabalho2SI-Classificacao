#include "Node.hpp"
#include <algorithm>
#include <numeric>

int Node::classificationOutput()
{
    // returns the most common class in outputsClassification
    if (outputsClassification.empty()) {
        return -1; // No classes available
    }

    std::vector<int> classCounts(*std::max_element(outputsClassification.begin(), outputsClassification.end()) + 1, 0);
    for (int cls : outputsClassification) {
        classCounts[cls]++;
    }
    int maxCount = classCounts[0];
    for (std::size_t i = 1; i < classCounts.size(); ++i) {
        if (classCounts[i] > classCounts[i-1]) {
            maxCount = classCounts[i];
        }
    }
    return maxCount;

}

double Node::regressionOutput()
{
    // average of outputsRegression
    if (outputsRegression.empty()) {
        return 0.0; // No values available
    }
    double sum = std::accumulate(outputsRegression.begin(), outputsRegression.end(), 0.0);
    return sum / outputsRegression.size();
}