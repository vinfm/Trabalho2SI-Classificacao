#ifndef CRITERIA_HPP
#define CRITERIA_HPP
#include <vector>
enum class CriterionType { Gini, MSE };

class Criteria {
public:
    Criteria(CriterionType type);

    static double gini(const std::vector<std::vector<double>> &data);
    static double mse(const std::vector<std::vector<double>> &data);
    double (*calculateImpurity)(const std::vector<std::vector<double>> &data);
};
#endif // CRITERIA_HPP