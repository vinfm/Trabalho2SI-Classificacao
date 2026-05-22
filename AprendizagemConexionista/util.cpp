#include "util.hpp"

void normalize_features(std::vector<std::vector<double>>& X,
                        double source_min,
                        double source_max,
                        double target_min,
                        double target_max) {
    if (X.empty() || X[0].empty()) return;

    const double source_range = source_max - source_min;
    const double target_mid = (target_min + target_max) / 2.0;

    if (source_range <= 1e-12) {
        for (std::size_t i = 0; i < X.size(); ++i) {
            for (std::size_t j = 0; j < X[i].size(); ++j) {
                X[i][j] = target_mid;
            }
        }
        return;
    }

    for (std::size_t i = 0; i < X.size(); ++i) {
        for (std::size_t j = 0; j < X[i].size(); ++j) {
            X[i][j] = target_min + (X[i][j] - source_min) * (target_max - target_min) / source_range;
        }
    }
}

void normalize_features(std::vector<std::vector<double>>& X,
                        const std::vector<double>& source_min,
                        const std::vector<double>& source_max,
                        double target_min,
                        double target_max) {
    if (X.empty() || X[0].empty()) return;
    if (source_min.size() != X[0].size() || source_max.size() != X[0].size()) return;

    const double target_mid = (target_min + target_max) / 2.0;

    for (std::size_t j = 0; j < X[0].size(); ++j) {
        const double source_range = source_max[j] - source_min[j];
        if (source_range <= 1e-12) {
            for (std::size_t i = 0; i < X.size(); ++i) {
                X[i][j] = target_mid;
            }
            continue;
        }

        for (std::size_t i = 0; i < X.size(); ++i) {
            X[i][j] = target_min + (X[i][j] - source_min[j]) * (target_max - target_min) / source_range;
        }
    }
}
