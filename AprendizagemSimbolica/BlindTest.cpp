#include "../Dataset.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <vector>

static void stratifiedSplit(const Dataset& full, double testFrac, unsigned seed,
                            Dataset& train, Dataset& test) {
    std::vector<int> classes = {1, 2, 3, 4};
    std::mt19937 rng(seed);
    train.X.clear(); train.Y.clear();
    test.X.clear();  test.Y.clear();
    for (int c : classes) {
        std::vector<std::size_t> idx;
        for (std::size_t i = 0; i < full.nrows(); ++i)
            if (full.Y[i].size() >= 2 && static_cast<int>(full.Y[i][1]) == c)
                idx.push_back(i);
        std::shuffle(idx.begin(), idx.end(), rng);
        std::size_t nTest = static_cast<std::size_t>(std::round(testFrac * idx.size()));
        for (std::size_t i = 0; i < idx.size(); ++i) {
            if (i < nTest) { test.X.push_back(full.X[idx[i]]); test.Y.push_back(full.Y[idx[i]]); }
            else           { train.X.push_back(full.X[idx[i]]); train.Y.push_back(full.Y[idx[i]]); }
        }
    }
}

int main() {
    Dataset full;
    if (!full.loadCSV("../02_treino_sinais_vitais_com_label.txt", {4, 5, 6})) {
        fprintf(stderr, "Nao foi possivel abrir o arquivo de treino\n");
        return 1;
    }

    Dataset train, test;
    stratifiedSplit(full, 0.30, 42u, train, test);

    // Arquivo cego (formato do enunciado: i, qPA, pulso, resp)
    std::ofstream cego("teste_cego_simulado.txt");
    for (std::size_t i = 0; i < test.nrows(); ++i) {
        cego << (i + 1) << ","
             << test.X[i][0] << ","   // qPA
             << test.X[i][1] << ","   // pulso
             << test.X[i][2] << "\n"; // resp
    }
    printf("Gerado teste_cego_simulado.txt com %zu linhas\n", test.nrows());

    // Gabarito (i, gravid, classe) -- pra voce comparar depois com predicoes.csv
    std::ofstream gab("teste_cego_gabarito.csv");
    gab << "i,gravid,classe\n";
    for (std::size_t i = 0; i < test.nrows(); ++i) {
        gab << (i + 1) << ","
            << test.Y[i][0] << ","
            << static_cast<int>(test.Y[i][1]) << "\n";
    }
    printf("Gerado teste_cego_gabarito.csv com %zu linhas\n", test.nrows());

    return 0;
}
