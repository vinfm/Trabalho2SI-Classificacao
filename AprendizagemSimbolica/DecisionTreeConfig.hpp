#ifndef DECISIONTREECONFIG_HPP
#define DECISIONTREECONFIG_HPP
#include <cstddef>
struct DecisionTreeConfig {
    std::size_t maxDepth = 5;
    std::size_t minSamplesLeaf = 10;
};
#endif