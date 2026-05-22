#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "NeuralNet.hpp"
#include "NeuralNetBuilder.hpp"
#include "../Dataset.hpp"
#include "../Metrics.hpp"
#include "util.hpp"

static void normalizeInputs(std::vector<std::vector<double>>& X) {
    if (X.empty()) {
        return;
    }
    static constexpr double SI3_MIN = -10.0;
    static constexpr double SI3_MAX = 10.0;
    static constexpr double SI4_MIN = 0.0;
    static constexpr double SI4_MAX = 200.0;
    static constexpr double SI5_MIN = 0.0;
    static constexpr double SI5_MAX = 22.0;
    static constexpr double INPUT_NORMALIZE_MIN = -1.0;
    static constexpr double INPUT_NORMALIZE_MAX = 1.0;
    std::vector<double> source_min = {SI3_MIN, SI4_MIN, SI5_MIN};
    std::vector<double> source_max = {SI3_MAX, SI4_MAX, SI5_MAX};
    normalize_features(X, source_min, source_max, INPUT_NORMALIZE_MIN, INPUT_NORMALIZE_MAX);
}

// Configurable parameters

static const Activation CLASSIFICATION_HIDDEN_LAYER_ACTIVATION = ACT_TANH;
static const Activation CLASSIFICATION_OUTPUT_LAYER_ACTIVATION = ACT_SOFTMAX;
static const Activation REGRESSION_HIDDEN_LAYER_ACTIVATION = ACT_TANH;
static const Activation REGRESSION_OUTPUT_LAYER_ACTIVATION = ACT_LINEAR;
static const std::vector<int> CLASS_LABELS = {1,2,3,4};
static const std::vector<int> TRAIN_INPUT_COLUMNS = {4,5,6};
static const std::vector<int> BLIND_INPUT_COLUMNS = {2,3,4};
static const std::vector<int> BLIND_LEGACY_COLUMNS = {4,5,6};
static constexpr double BLIND_RELOAD_THRESHOLD = 10.5;
static constexpr int NETWORK_LAYER_COUNT = 2;
static const std::vector<int> CLASSIFICATION_NEURONS_PER_LAYER = {12, 4};
static const std::vector<int> REGRESSION_NEURONS_PER_LAYER = {8, 1};
static constexpr int EPOCHS = 100;
static constexpr int BATCH_SIZE = 8;
static constexpr float CLASSIFICATION_LEARNING_RATE = 0.01f;
static constexpr float REGRESSION_LEARNING_RATE = 0.00001f;
static constexpr float CLASSIFICATION_MOMENTUM = 0.9f;
static constexpr float REGRESSION_MOMENTUM = 0.0f;
static constexpr float MIN_ERROR = 1e-6f;
static constexpr float VALIDATION_FRACTION = 0.2f;
static constexpr double TRAIN_TEST_FRACTION = 0.30;

static void stratifiedSplit(const Dataset& full, double testFrac, unsigned seed, Dataset& train, Dataset& test) {
    std::vector<int> classes = CLASS_LABELS;
    std::mt19937 rng(seed);

    train.X.clear();
    train.Y.clear();
    test.X.clear();
    test.Y.clear();

    for (int c : classes) {
        std::vector<std::size_t> idx;
        for (std::size_t i = 0; i < full.nrows(); ++i) {
            if (full.Y[i].size() >= 2 && static_cast<int>(full.Y[i][1]) == c) {
                idx.push_back(i);
            }
        }
        std::shuffle(idx.begin(), idx.end(), rng);
        std::size_t nTest = static_cast<std::size_t>(std::round(testFrac * idx.size()));
        for (std::size_t i = 0; i < idx.size(); ++i) {
            if (i < nTest) {
                test.X.push_back(full.X[idx[i]]);
                test.Y.push_back(full.Y[idx[i]]);
            } else {
                train.X.push_back(full.X[idx[i]]);
                train.Y.push_back(full.Y[idx[i]]);
            }
        }
    }
}

static Dataset projectY(const Dataset& src, std::size_t col) {
    Dataset out;
    out.X = src.X;
    out.Y.reserve(src.nrows());
    for (const auto& y : src.Y) {
        std::vector<double> row;
        if (col < y.size()) {
            row.push_back(y[col]);
        }
        out.Y.push_back(std::move(row));
    }
    return out;
}

static Dataset toOneHot(const Dataset& src, const std::vector<int>& classes) {
    Dataset out;
    out.X = src.X;
    out.Y.reserve(src.nrows());
    for (const auto& y : src.Y) {
        std::vector<double> row(classes.size(), 0.0);
        if (!y.empty()) {
            int c = static_cast<int>(y[0]);
            auto it = std::find(classes.begin(), classes.end(), c);
            if (it != classes.end()) {
                std::size_t idx = static_cast<std::size_t>(std::distance(classes.begin(), it));
                row[idx] = 1.0;
            }
        }
        out.Y.push_back(std::move(row));
    }
    return out;
}

static bool loadBlind(const std::string& path, Dataset& blind) {
    if (!blind.loadCSV(path, BLIND_INPUT_COLUMNS) || blind.nrows() == 0) {
        std::cerr << "Falha ao abrir " << path << "\n";
        return false;
    }

    bool outOfRange = false;
    if (!blind.X.empty()) {
        double v = blind.X[0][0];
        if (v < -BLIND_RELOAD_THRESHOLD || v > BLIND_RELOAD_THRESHOLD) {
            outOfRange = true;
        }
    }

    if (outOfRange) {
        blind = Dataset();
        if (!blind.loadCSV(path, BLIND_LEGACY_COLUMNS) || blind.nrows() == 0) {
            std::cerr << "Falha ao reabrir " << path << " em formato legado\n";
            return false;
        }
    }

    return true;
}

static void writePredictions(const std::string& path,
                             NeuralNet& regNet,
                             NeuralNet& clfNet,
                             const Dataset& X) {
    std::ofstream out(path);
    out << "i,gravid,classe\n";

    for (std::size_t i = 0; i < X.nrows(); ++i) {
        std::vector<double> r = regNet.predict(X.X[i]);
        double g = r.empty() ? 0.0 : r[0];

        std::vector<double> c = clfNet.predict(X.X[i]);
        int cls = 1;
        if (!c.empty()) {
            auto it = std::max_element(c.begin(), c.end());
            cls = static_cast<int>(std::distance(c.begin(), it)) + 1;
        }

        out << (i + 1) << "," << g << "," << cls << "\n";
    }

    printf("  -> %s (%zu linhas)\n", path.c_str(), X.nrows());
}

int main(int argc, char** argv) {
    std::string trainPath = (argc >= 2) ? argv[1] : "../02_treino_sinais_vitais_com_label.txt";
    std::string blindPath = (argc >= 3) ? argv[2] : "../01_treino_sinais_vitais_sem_label.txt";
    unsigned seed = (argc >= 4) ? static_cast<unsigned>(std::stoul(argv[3])) : 42u;

    Dataset full;
    if (!full.loadCSV(trainPath, TRAIN_INPUT_COLUMNS)) {
        std::cerr << "Nao foi possivel abrir " << trainPath << "\n";
        return 1;
    }
    printf("Linhas carregadas: %zu\n", full.nrows());

    Dataset train, test;
    stratifiedSplit(full, TRAIN_TEST_FRACTION, seed, train, test);
    printf("Treino: %zu  |  Teste: %zu\n", train.nrows(), test.nrows());

    Dataset trainReg = projectY(train, 0);
    Dataset trainClf = projectY(train, 1);
    Dataset testReg = projectY(test, 0);
    Dataset testClf = projectY(test, 1);

    const std::size_t nFeat = train.ninputs();
    normalizeInputs(train.X);
    normalizeInputs(test.X);

    Dataset trainClfOneHot = toOneHot(trainClf, CLASS_LABELS);

    trainReg.X = train.X;
    testReg.X = test.X;
    trainClfOneHot.X = train.X;
    testClf.X = test.X;

    const int epochs = EPOCHS;
    const int batch_size = BATCH_SIZE;
    const float learning_rate = CLASSIFICATION_LEARNING_RATE;
    const float regression_learning_rate = REGRESSION_LEARNING_RATE;
    const float momentum = CLASSIFICATION_MOMENTUM;
    const float regression_momentum = REGRESSION_MOMENTUM;
    const float min_error = MIN_ERROR;
    const float test_fraction = VALIDATION_FRACTION;

    printf("Treinando rede (Regressao)\n");
    NeuralNetBuilder builderReg(trainReg);
    NeuralNet netReg = builderReg.build_net(NETWORK_LAYER_COUNT, REGRESSION_NEURONS_PER_LAYER, static_cast<int>(nFeat));
    builderReg.set_parameters(netReg, REGRESSION_MOMENTUM, REGRESSION_LEARNING_RATE, 0.0f, BATCH_SIZE, VALIDATION_FRACTION, MIN_ERROR);
    netReg.set_activation(REGRESSION_HIDDEN_LAYER_ACTIVATION);
    netReg.set_activation_output_layer(REGRESSION_OUTPUT_LAYER_ACTIVATION);
    builderReg.split_dataset(seed);
    builderReg.train_net(netReg, EPOCHS, seed);

    printf("Treinando rede (Classificacao)\n");
    NeuralNetBuilder builderClf(trainClfOneHot);
    NeuralNet netClf = builderClf.build_net(NETWORK_LAYER_COUNT, CLASSIFICATION_NEURONS_PER_LAYER, static_cast<int>(nFeat));
    
    builderClf.set_parameters(netClf, CLASSIFICATION_MOMENTUM, CLASSIFICATION_LEARNING_RATE, 0.0f, BATCH_SIZE, VALIDATION_FRACTION, MIN_ERROR);
    netClf.set_activation(CLASSIFICATION_HIDDEN_LAYER_ACTIVATION);
    netClf.set_activation_output_layer(CLASSIFICATION_OUTPUT_LAYER_ACTIVATION);
    builderClf.split_dataset(seed);
    builderClf.train_net(netClf, EPOCHS, seed);

    std::vector<double> gTrue;
    std::vector<int> cTrue;
    std::vector<double> gPredNet;
    std::vector<int> cPredNet;

    for (std::size_t i = 0; i < test.X.size(); ++i) {
        double g_orig = testReg.Y[i].empty() ? 0.0 : testReg.Y[i][0];
        gTrue.push_back(g_orig);

        int cls_true = static_cast<int>(testClf.Y[i][0]);
        cTrue.push_back(cls_true);

        std::vector<double> rp = netReg.predict(test.X[i]);
        double gpred_orig = rp.empty() ? 0.0 : rp[0];
        gPredNet.push_back(gpred_orig);

        std::vector<double> cp = netClf.predict(test.X[i]);
        int pred_cls = 1;
        if (!cp.empty()) {
            pred_cls = static_cast<int>(std::distance(cp.begin(), std::max_element(cp.begin(), cp.end()))) + 1;
        }
        cPredNet.push_back(pred_cls);
    }

    printf("#              RESULTADOS NO TESTE                #\n");
    printf("\nRegressao (gravidade)\n");
    metrics::printRegressionReport(gTrue, gPredNet, "NeuralNet");
    printf("\nClassificacao (classe)\n");
    metrics::printClassificationReport(cTrue, cPredNet, CLASS_LABELS, "NeuralNet");
    printf("\nComparacao resumida\n");
    printf("RMSE (gravidade)    %8.4f\n", metrics::rmse(gTrue, gPredNet));
    printf("MAE  (gravidade)    %8.4f\n", metrics::mae(gTrue, gPredNet));
    printf("Acuracia (classe)   %8.4f\n", metrics::accuracy(cTrue, cPredNet));
    auto cmNet = metrics::confusionMatrix(cTrue, cPredNet, CLASS_LABELS);
    printf("F1 macro (classe)   %8.4f\n", metrics::f1Macro(cmNet));

    Dataset blind;
    if (!loadBlind(blindPath, blind)) {
        return 1;
    }
    normalizeInputs(blind.X);
    printf("Arquivo de teste cego: %s\n", blindPath.c_str());
    printf("Amostras a prever: %zu\n", blind.nrows());

    writePredictions("predicoes_rede.csv", netReg, netClf, blind);
    return 0;
}