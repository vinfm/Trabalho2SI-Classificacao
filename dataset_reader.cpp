#include "Dataset.hpp"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [<train_out> <test_out> <full_out>]\n";
        return 1;
    }

    std::string inpath = argv[1];
    // Use the real features, not the example id column.
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

    std::string trainFile = "train_output.csv";
    std::string testFile = "test_output.csv";
    std::string fullFile = "full_output.csv";

    if (argc >= 3) {
        trainFile = argv[2];
    }
    if (argc >= 4) {
        testFile = argv[3];
    }
    if (argc >= 5) {
        fullFile = argv[4];
    }

    if (!ds.writeCSV(fullFile)) {
        std::cerr << "Failed to write full output file.\n";
        return 1;
    }

    Dataset train;
    Dataset test;
    ds.split(0.5, 0, train, test);

    if (!train.writeCSV(trainFile) || !test.writeCSV(testFile)) {
        std::cerr << "Failed to write train/test output files.\n";
        return 1;
    }

    std::cout << "# Train rows=" << train.nrows() << "\n";
    std::cout << "# Test rows=" << test.nrows() << "\n";
    std::cout << "Wrote train/test split and full ordered CSV: " << fullFile << "\n";

    return 0;
}
