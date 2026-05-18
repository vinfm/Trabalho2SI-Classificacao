#ifndef TREEBUILDER_HPP
#define TREEBUILDER_HPP
#include "DecisionTree.hpp"
#include <vector>
enum class TreeType { Classification, Regression };
class TreeBuilder {
public:

    TreeBuilder();
    DecisionTree* CARTBuild(const std::vector<std::vector<double>>& data);
    std::pair<int, double> findBestSplit(const std::vector<std::vector<double>>& data);
private:
    
};
#endif // TREEBUILDER_HPP