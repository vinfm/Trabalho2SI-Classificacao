#include "Criteria.hpp"
#include <algorithm>
#include <numeric>

Criteria::Criteria(CriterionType type)
{
    switch (type) {
    case CriterionType::Gini:
        calculateImpurity = gini;
        break;
    case CriterionType::MSE:
        calculateImpurity = mse;
        break;
    default:
        calculateImpurity = nullptr;
        break;
    }
}

double Criteria::gini(const std::vector<double> &data)
{
    std::vector<int> classCounts(*std::max_element(data.begin(), data.end()) + 1, 0);
    for (double cls : data) {
        classCounts[static_cast<int>(cls)]++;
    }
    double freq_squares = 0.0;
    for (int count : classCounts) {
        freq_squares += count * count;
    }
    return 1.0 - freq_squares / (data.size() * data.size());
}

double Criteria::mse(const std::vector<double> &data)
{
    if (data.empty()) return 0.0;
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double mse = 0.0;
    for (double val : data) {
        mse += (val - mean) * (val - mean);
    }
    return mse / data.size();
}

void Criteria::NodeImpurity(double &impurity, const std::vector<double> &data)
{
    impurity = calculateImpurity(data);
}
