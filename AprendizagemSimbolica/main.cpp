#include "Forest.hpp"
#include "ForestBuilder.hpp"
#include "../Dataset.hpp"
#include "Metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// Divide os dados por classe para manter a proporção no treino e no teste
static void stratifiedSplit(const Dataset& full, double testFrac, unsigned seed,
                            Dataset& train, Dataset& test) {
    std::vector<int> classes = {1, 2, 3, 4};
    std::mt19937 rng(seed);

    train.X.clear(); train.Y.clear();
    test.X.clear();  test.Y.clear();

    for (int c : classes) {
        std::vector<std::size_t> idx;
        for (std::size_t i = 0; i < full.nrows(); ++i) {
            if (full.Y[i].size() >= 2 && static_cast<int>(full.Y[i][1]) == c) idx.push_back(i);
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

// Projeta as colunas Y para um subconjunto (ex: {0} = só gravidade, {1} = só classe)
static Dataset projectY(const Dataset& src, std::size_t col) {
    Dataset out;
    out.X = src.X;
    out.Y.reserve(src.nrows());
    for (const auto& y : src.Y) {
        std::vector<double> row;
        if (col < y.size()) row.push_back(y[col]);
        out.Y.push_back(std::move(row));
    }
    return out;
}

int main(int argc, char** argv) {
    std::string trainPath = (argc >= 2) ? argv[1] : "../02_treino_sinais_vitais_com_label.txt";
    std::string blindPath = (argc >= 3) ? argv[2] : "";
    int  nTrees    = (argc >= 4) ? std::stoi(argv[3]) : 100;
    int  maxDepth  = (argc >= 5) ? std::stoi(argv[4]) : 15;
    unsigned seed  = (argc >= 6) ? static_cast<unsigned>(std::stoul(argv[5])) : 42u;

    // Carrega os dados de treino
    // cols 4,5,6 são as features e Y guarda [gravidade, classe]
    Dataset full;
    if (!full.loadCSV(trainPath, {4, 5, 6})) {
        std::cerr << "Não foi possível abrir " << trainPath << "\n";
        return 1;
    }
    printf("Foram carregadas %zu linhas\n", full.nrows());

    // Separa em treino e teste mantendo as classes balanceadas
    Dataset train, test;
    stratifiedSplit(full, 0.30, seed, train, test);
    printf("Treino: %zu, Teste: %zu\n", train.nrows(), test.nrows());

    // Prepara os dados para cada tarefa: regressão e classificação
    Dataset trainReg = projectY(train, 0);  // só gravidade
    Dataset trainClf = projectY(train, 1);  // só classe

    // Treina o modelo de regressão
    printf("\nTreinando modelo de regressão\n");
    ForestBuilder regBuilder(TreeType::Regression, &trainReg, &trainReg,
                             /*min_samples_leaf*/2, maxDepth,
                             /*min_samples_split*/5, /*min_impurity*/1e-7,
                             /*n_features*/3, nTrees);
    Forest* regForest = regBuilder.buildForest();

    // Treina o modelo de classificação
    printf("Treinando modelo de classificação\n");
    ForestBuilder clfBuilder(TreeType::Classification, &trainClf, &trainClf,
                             2, maxDepth, 5, 1e-7, 3, nTrees);
    Forest* clfForest = clfBuilder.buildForest();

    // Avalia os modelos no conjunto de teste
    std::vector<double> gTrue, gPred;
    std::vector<int>    cTrue, cPred;
    for (std::size_t i = 0; i < test.nrows(); ++i) {
        gTrue.push_back(test.Y[i][0]);
        cTrue.push_back(static_cast<int>(test.Y[i][1]));
        gPred.push_back(regForest->predictRegression(test.X[i]));
        cPred.push_back(clfForest->predictClassification(test.X[i]));
    }
    metrics::printRegressionReport(gTrue, gPred, "Teste");
    metrics::printClassificationReport(cTrue, cPred, {1, 2, 3, 4}, "Teste");

    // Se houver arquivo cego, faz previsões e salva em CSV
    if (!blindPath.empty()) {
        Dataset blind;
        if (!blind.loadCSV(blindPath, {2, 3, 4}) || blind.nrows() == 0) {
            std::cerr << "Não foi possível abrir o teste cego: " << blindPath << "\n";
            delete regForest; delete clfForest;
            return 1;
        }
        // Se o formato não for o esperado, tenta carregar o arquivo no segundo formato
        bool outOfRange = false;
        if (!blind.X.empty()) {
            double v = blind.X[0][0];
            if (v < -10.5 || v > 10.5) outOfRange = true;
        }
        if (outOfRange) {
            blind = Dataset();
            blind.loadCSV(blindPath, {4, 5, 6});
        }

        std::ofstream out("predicoes.csv");
        out << "i,gravid,classe\n";
        for (std::size_t i = 0; i < blind.nrows(); ++i) {
            double g = regForest->predictRegression(blind.X[i]);
            int    c = clfForest->predictClassification(blind.X[i]);
            out << (i + 1) << "," << g << "," << c << "\n";
        }
        printf("\nPredições salvas em predicoes.csv (%zu linhas)\n", blind.nrows());
    }

    delete regForest;
    delete clfForest;
    return 0;
}