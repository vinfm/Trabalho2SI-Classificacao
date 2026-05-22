#ifndef UTIL_HPP
#define UTIL_HPP

#include <cstddef>
#include <vector>

void normalize_features(std::vector<std::vector<double>>& X,
                        double source_min,
                        double source_max,
                        double target_min,
                        double target_max);

void normalize_features(std::vector<std::vector<double>>& X,
                        const std::vector<double>& source_min,
                        const std::vector<double>& source_max,
                        double target_min,
                        double target_max);

#endif