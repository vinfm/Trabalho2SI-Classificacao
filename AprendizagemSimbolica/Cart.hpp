#ifndef CART_HPP
#define CART_HPP

#include <cstddef>
#include <string>
#include <vector>
#include "Node.hpp"
#include "Dataset.hpp"
#include "DecisionTree.hpp"
class CART : public DecisionTree {
    public:
        CART() = default;
        ~CART() = default;
        void saveModel(const std::string &path) const;
        void loadModel(const std::string &path);
        void printTree() const;
        int predictClassification(const std::vector<double> &features) = delete;
        double predictRegression(const std::vector<double> &features) = 0;
};
#endif