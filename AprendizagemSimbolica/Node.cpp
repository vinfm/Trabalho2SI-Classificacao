#include "Node.hpp"
#include <algorithm>
#include <numeric>

int Node::classificationOutput()
{
    // returns the most common class in valuesClassification
    if (valuesClassification.empty()) {
        return -1; // No classes available
    }

    std::vector<int> classCounts(*std::max_element(valuesClassification.begin(), valuesClassification.end()) + 1, 0);
    for (int cls : valuesClassification) {
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
    // average of valuesRegression
    if (valuesRegression.empty()) {
        return 0.0; // No values available
    }
    double sum = std::accumulate(valuesRegression.begin(), valuesRegression.end(), 0.0);
    return sum / valuesRegression.size();
}