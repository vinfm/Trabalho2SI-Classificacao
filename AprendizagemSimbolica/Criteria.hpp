#ifndef CRITERIA_HPP
#define CRITERIA_HPP
#include <vector>
enum class CriterionType { Gini, MSE };

class Criteria {
public:
    Criteria(CriterionType type);
    static double gini(const std::vector<double> &data);
    static double mse(const std::vector<double> &data);
    double (*calculateImpurity)(const std::vector<double> &data);
    void ChildrenImpurities(double &impurity_left, double &impurity_right, const std::vector<double> &data_left, const std::vector<double> &data_right);
    void NodeImpurity(double &impurity, const std::vector<double> &data);
};
#endif // CRITERIA_HPP