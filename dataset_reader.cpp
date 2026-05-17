#include "Dataset.hpp"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [<train_out> <test_out>]\n";
        return 1;
    }

    std::string inpath = argv[1];
    std::vector<int> inputCols = {1, 4, 5, 6}; // 1-based

    Dataset ds;
    if (!ds.loadCSV(inpath, inputCols)) {
        std::cerr << "Failed to load file: " << inpath << "\n";
        return 1;
    }

    if (ds.nrows() == 0) {
        std::cerr << "No valid rows parsed.\n";
        return 1;
    }

    // Neural networks benefit from normalized inputs.
    ds.standardize();

    Dataset train;
    Dataset test;
    ds.split(0.5, 0, train, test);

    if (argc >= 4) {
        std::string trainFile = argv[2];
        std::string testFile = argv[3];

        if (!train.writeCSV(trainFile) || !test.writeCSV(testFile)) {
            std::cerr << "Failed to write output files.\n";
            return 1;
        }

        std::cout << "Wrote " << train.nrows() << " train and " << test.nrows() << " test rows\n";
    } else {
        if (!train.writeCSV("train_output.csv") || !test.writeCSV("test_output.csv")) {
            std::cerr << "Failed to write train_output.csv or test_output.csv.\n";
            return 1;
        }

        std::cout << "# Train rows=" << train.nrows() << "\n";
        std::cout << "# Test rows=" << test.nrows() << "\n";
        std::cout << "Wrote train_output.csv and test_output.csv in working dir.\n";
    }

    return 0;
}
