#ifndef METRICS_HPP
#define METRICS_HPP

#include <vector>
#include <string>

namespace metrics {

double rmse(const std::vector<double>& yTrue, const std::vector<double>& yPred);
double mae(const std::vector<double>& yTrue, const std::vector<double>& yPred);
double accuracy(const std::vector<int>& yTrue, const std::vector<int>& yPred);
std::vector<std::vector<int>> confusionMatrix(const std::vector<int>& yTrue, const std::vector<int>& yPred, const std::vector<int>& classes);
double precisionMacro(const std::vector<std::vector<int>>& cm);
double recallMacro(const std::vector<std::vector<int>>& cm);
double f1Macro(const std::vector<std::vector<int>>& cm);
void printRegressionReport(const std::vector<double>& yTrue, const std::vector<double>& yPred, const std::string& title);
void printClassificationReport(const std::vector<int>& yTrue, const std::vector<int>& yPred, const std::vector<int>& classes, const std::string& title);

} 

#endif