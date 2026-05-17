#include "Dataset.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>

namespace {
std::vector<std::string> splitLine(const std::string &s, char delim) {
    std::vector<std::string> elems;
    std::string cur;
    std::istringstream iss(s);
    while (std::getline(iss, cur, delim)) {
        elems.push_back(cur);
    }
    return elems;
}

std::string trim(const std::string &s) {
    std::size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return std::string();
    }
    std::size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}
}

std::size_t Dataset::nrows() const {
    return X.size();
}

std::size_t Dataset::ninputs() const {
    return X.empty() ? 0 : X[0].size();
}

std::size_t Dataset::noutputs() const {
    return Y.empty() ? 0 : Y[0].size();
}

bool Dataset::loadCSV(const std::string &path, const std::vector<int> &inputCols) {
    if (inputCols.empty()) {
        return false;
    }

    std::ifstream fi(path);
    if (!fi) {
        return false;
    }

    int maxInput = *std::max_element(inputCols.begin(), inputCols.end());

    std::string line;
    while (std::getline(fi, line)) {
        if (line.empty()) {
            continue;
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        std::vector<std::string> cols = splitLine(line, ',');
        for (std::string &c : cols) {
            c = trim(c);
        }
        if (static_cast<int>(cols.size()) < maxInput) {
            continue;
        }

        std::vector<double> xin;
        bool ok = true;
        for (int idx : inputCols) {
            try {
                xin.push_back(std::stod(cols[idx - 1]));
            } catch (...) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        std::vector<double> yout;
        for (int i = maxInput; i < static_cast<int>(cols.size()); ++i) {
            try {
                yout.push_back(std::stod(cols[i]));
            } catch (...) {
                yout.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        }

        X.push_back(std::move(xin));
        Y.push_back(std::move(yout));
    }

    return true;
}

bool Dataset::writeCSV(const std::string &path) const {
    std::ofstream fo(path);
    if (!fo) {
        return false;
    }

    std::size_t ni = ninputs();
    std::size_t no = noutputs();

    for (std::size_t i = 0; i < ni; ++i) {
        if (i) {
            fo << ',';
        }
        fo << "in" << (i + 1);
    }
    for (std::size_t j = 0; j < no; ++j) {
        fo << ",out" << (j + 1);
    }
    fo << '\n';

    for (std::size_t r = 0; r < nrows(); ++r) {
        for (std::size_t i = 0; i < ni; ++i) {
            if (i) {
                fo << ',';
            }
            fo << X[r][i];
        }
        for (std::size_t j = 0; j < no; ++j) {
            double value = std::numeric_limits<double>::quiet_NaN();
            if (j < Y[r].size() && !std::isnan(Y[r][j])) {
                value = Y[r][j];
            }
            fo << ',' << value;
        }
        fo << '\n';
    }

    return true;
}

void Dataset::split(double test_frac, unsigned seed, Dataset &train, Dataset &test) const {
    train.X.clear();
    train.Y.clear();
    test.X.clear();
    test.Y.clear();

    std::vector<int> idx(nrows());
    std::iota(idx.begin(), idx.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    std::size_t ntest = static_cast<std::size_t>(std::round(test_frac * nrows()));
    for (std::size_t i = 0; i < idx.size(); ++i) {
        if (i < ntest) {
            test.X.push_back(X[idx[i]]);
            if (!Y.empty()) {
                test.Y.push_back(Y[idx[i]]);
            }
        } else {
            train.X.push_back(X[idx[i]]);
            if (!Y.empty()) {
                train.Y.push_back(Y[idx[i]]);
            }
        }
    }
}

void Dataset::standardize() {
    if (X.empty()) {
        return;
    }

    std::size_t n = X.size();
    std::size_t m = X[0].size();
    std::vector<double> mean(m, 0.0);
    std::vector<double> sd(m, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            mean[j] += X[i][j];
        }
    }
    for (std::size_t j = 0; j < m; ++j) {
        mean[j] /= static_cast<double>(n);
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            double d = X[i][j] - mean[j];
            sd[j] += d * d;
        }
    }
    for (std::size_t j = 0; j < m; ++j) {
        sd[j] = std::sqrt(sd[j] / static_cast<double>(n));
        if (sd[j] == 0.0) {
            sd[j] = 1.0;
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            X[i][j] = (X[i][j] - mean[j]) / sd[j];
        }
    }
}
