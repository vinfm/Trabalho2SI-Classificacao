#ifndef DATASET_HPP
#define DATASET_HPP

#include <cstddef>
#include <string>
#include <vector>

class Dataset {
public:
    // X: rows x features, Y: rows x outputs
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> Y;

    Dataset() = default;
    Dataset(std::size_t nrows, std::size_t ninputs) {
        X.resize(nrows, std::vector<double>(ninputs, 0.0));
        Y.resize(nrows); // outputs can be variable-length
    }
    std::size_t nrows() const;
    std::size_t ninputs() const;
    std::size_t noutputs() const;

    // Load CSV: inputCols are 1-based indices. Columns after max(inputCols) become outputs.
    bool loadCSV(const std::string &path, const std::vector<int> &inputCols);

    // Write dataset to CSV with header in1..inN,out1..outM
    bool writeCSV(const std::string &path) const;

    // Random train/test split by fraction (test_frac in [0,1]). Seed for reproducibility.
    void split(double test_frac, unsigned seed, Dataset &train, Dataset &test) const;

    // Standardize inputs (zero mean, unit variance). Operates in-place on X.
    void standardize();
};

#endif
