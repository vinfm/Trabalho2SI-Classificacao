#include "Criteria.hpp"

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

double Criteria::gini(const std::vector<std::vector<double>> &data)
{
    return 0.0;
}

double Criteria::mse(const std::vector<std::vector<double>> &data)
{
    return 0.0;
}