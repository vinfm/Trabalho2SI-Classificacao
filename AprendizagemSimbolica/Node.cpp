#include "Node.hpp"
#include <algorithm>
#include <numeric>

int Node::classificationOutput()
{
    if (valuesClassification.empty()) {
        return -1;
    }

    int maxClass = *std::max_element(valuesClassification.begin(), valuesClassification.end());
    std::vector<int> classCounts(maxClass + 1, 0);
    for (int cls : valuesClassification) {
        classCounts[cls]++;
    }

    int bestClass = 0;
    int bestCount = -1;
    for (std::size_t i = 0; i < classCounts.size(); ++i) {
        if (classCounts[i] > bestCount) {
            bestCount = classCounts[i];
            bestClass = static_cast<int>(i);
        }
    }
    return bestClass;
}

double Node::regressionOutput()
{
    // average of valuesRegression
    if (valuesRegression.empty()) {
        return 0.0; // No values available
    }
    double sum = std::accumulate(valuesRegression.begin(), valuesRegression.end(), 0.0);
    return sum / valuesRegression.size();
}