#ifndef SPLITTER_HPP
#define SPLITTER_HPP
#include <vector>
class Splitter {
public:
    Splitter();
    std::pair<int, double> findBestSplit(const std::vector<std::vector<double>> &data);
private:
};
#endif