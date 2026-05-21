#include "Forest.hpp"
#include "ForestBuilder.hpp"
#include "TreeBuilder.hpp"
#include "Cart.hpp"
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

// ============================================================
// Divide os dados por classe para manter a proporcao no treino e no teste
// ============================================================
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

// Projeta as colunas Y para um subconjunto (col 0 = gravidade, col 1 = classe)
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

// ============================================================
// Treina um CART unico (sem bootstrap, sem random subspace) e
// envelopa numa "Forest" de 1 arvore para reusar predict do Forest.
// ============================================================
static Forest* buildSingleCART(TreeType type, Dataset& trainProj,
                               int minLeaf, int maxDepth, int minSplit,
                               double minImpurity, unsigned seed) {
    std::size_t nfeat = trainProj.ninputs();
    TreeBuilder* tb = new TreeBuilder(type, &trainProj, &trainProj,
                                      minLeaf, maxDepth, minSplit, minImpurity,
                                      nfeat);
    // forca max_features = n_features (sem random subspace) -> arvore deterministica
    tb->setMaxFeatures(nfeat);
    tb->setRandomSeed(seed);
    DecisionTree* tree = tb->CARTBuild();
    delete tb;

    Forest* f = new Forest(1);
    f->addTree(tree);
    return f;
}

// ============================================================
// Detecta formato do arquivo cego e carrega adequadamente.
// Formato A (enunciado puro):  i, qPA, pulso, resp                (cols 2,3,4)
// Formato B (legado/dataset):  i, pSist, pDiast, qPA, pulso, resp [, gravid]  (cols 4,5,6)
// ============================================================
static bool loadBlind(const std::string& path, Dataset& blind) {
    // Primeiro tenta o formato do enunciado (4 colunas)
    if (!blind.loadCSV(path, {2, 3, 4}) || blind.nrows() == 0) {
        std::cerr << "Falha ao abrir " << path << "\n";
        return false;
    }
    // Se qPA ficar fora de [-10,10], foi parseado errado -> tenta formato legado
    bool outOfRange = false;
    if (!blind.X.empty()) {
        double v = blind.X[0][0];
        if (v < -10.5 || v > 10.5) outOfRange = true;
    }
    if (outOfRange) {
        blind = Dataset();
        if (!blind.loadCSV(path, {4, 5, 6}) || blind.nrows() == 0) {
            std::cerr << "Falha ao reabrir " << path << " em formato legado\n";
            return false;
        }
    }
    return true;
}

// ============================================================
// Escreve CSV de predicoes com cabecalho i,gravid,classe
// ============================================================
static void writePredictions(const std::string& path,
                             Forest* reg, Forest* clf,
                             const Dataset& X) {
    std::ofstream out(path);
    out << "i,gravid,classe\n";
    for (std::size_t i = 0; i < X.nrows(); ++i) {
        double g = reg->predictRegression(X.X[i]);
        int    c = clf->predictClassification(X.X[i]);
        out << (i + 1) << "," << g << "," << c << "\n";
    }
    printf("  -> %s (%zu linhas)\n", path.c_str(), X.nrows());
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char** argv) {
    std::string trainPath = (argc >= 2) ? argv[1] : "../02_treino_sinais_vitais_com_label.txt";
    std::string blindPath = (argc >= 3) ? argv[2] : "";
    int  nTrees    = (argc >= 4) ? std::stoi(argv[3]) : 100;
    int  maxDepth  = (argc >= 5) ? std::stoi(argv[4]) : 15;
    unsigned seed  = (argc >= 6) ? static_cast<unsigned>(std::stoul(argv[5])) : 42u;

    // -----------------------------------------------------------------
    // 1) Carrega dados de treino completos
    //    cols 4,5,6 = qPA, pulso, resp (features)
    //    Y guarda [gravidade, classe] (cols restantes)
    // -----------------------------------------------------------------
    Dataset full;
    if (!full.loadCSV(trainPath, {4, 5, 6})) {
        std::cerr << "Nao foi possivel abrir " << trainPath << "\n";
        return 1;
    }
    printf("Linhas carregadas: %zu\n", full.nrows());

    // -----------------------------------------------------------------
    // 2) Holdout estratificado 70/30
    // -----------------------------------------------------------------
    Dataset train, test;
    stratifiedSplit(full, 0.30, seed, train, test);
    printf("Treino: %zu  |  Teste: %zu\n", train.nrows(), test.nrows());

    Dataset trainReg = projectY(train, 0);  // gravidade
    Dataset trainClf = projectY(train, 1);  // classe

    // hiperparametros compartilhados
    const int minLeaf    = 2;
    const int minSplit   = 5;
    const double minImp  = 1e-7;
    const std::size_t nFeat = 3;

    // =================================================================
    // TECNICA 1: CART UNICO (arvore de decisao classica)
    // =================================================================
    printf("\n===========================================\n");
    printf(" TECNICA 1: CART UNICO\n");
    printf("===========================================\n");

    printf("Treinando CART de regressao...\n");
    Forest* cartReg = buildSingleCART(TreeType::Regression, trainReg,
                                      minLeaf, maxDepth, minSplit, minImp, seed);

    printf("Treinando CART de classificacao...\n");
    Forest* cartClf = buildSingleCART(TreeType::Classification, trainClf,
                                      minLeaf, maxDepth, minSplit, minImp, seed);

    // =================================================================
    // TECNICA 2: RANDOM FOREST (ensemble com bootstrap + random subspace)
    // =================================================================
    printf("\n===========================================\n");
    printf(" TECNICA 2: RANDOM FOREST (%d arvores)\n", nTrees);
    printf("===========================================\n");

    printf("Treinando Random Forest de regressao...\n");
    ForestBuilder rfRegBuilder(TreeType::Regression, &trainReg, &trainReg,
                               minLeaf, maxDepth, minSplit, minImp, nFeat, nTrees);
    Forest* rfReg = rfRegBuilder.buildForest();

    printf("Treinando Random Forest de classificacao...\n");
    ForestBuilder rfClfBuilder(TreeType::Classification, &trainClf, &trainClf,
                               minLeaf, maxDepth, minSplit, minImp, nFeat, nTrees);
    Forest* rfClf = rfClfBuilder.buildForest();

    // =================================================================
    // AVALIACAO NO TESTE (split interno honesto)
    // =================================================================
    std::vector<double> gTrue;
    std::vector<int>    cTrue;
    std::vector<double> gPredCART, gPredRF;
    std::vector<int>    cPredCART, cPredRF;

    for (std::size_t i = 0; i < test.nrows(); ++i) {
        gTrue.push_back(test.Y[i][0]);
        cTrue.push_back(static_cast<int>(test.Y[i][1]));
        gPredCART.push_back(cartReg->predictRegression(test.X[i]));
        gPredRF  .push_back(rfReg  ->predictRegression(test.X[i]));
        cPredCART.push_back(cartClf->predictClassification(test.X[i]));
        cPredRF  .push_back(rfClf  ->predictClassification(test.X[i]));
    }

    printf("\n###################################################\n");
    printf("#              RESULTADOS NO TESTE                #\n");
    printf("###################################################\n");

    printf("\n----- REGRESSAO (gravidade) -----");
    metrics::printRegressionReport(gTrue, gPredCART, "CART unico");
    metrics::printRegressionReport(gTrue, gPredRF,   "Random Forest");

    printf("\n----- CLASSIFICACAO (classe 1..4) -----");
    metrics::printClassificationReport(cTrue, cPredCART, {1,2,3,4}, "CART unico");
    metrics::printClassificationReport(cTrue, cPredRF,   {1,2,3,4}, "Random Forest");

    // Tabela comparativa resumida
    printf("\n=== COMPARACAO RESUMIDA ===\n");
    printf("                       CART        RF\n");
    printf("RMSE (gravidade)    %8.4f  %8.4f\n",
           metrics::rmse(gTrue, gPredCART), metrics::rmse(gTrue, gPredRF));
    printf("MAE  (gravidade)    %8.4f  %8.4f\n",
           metrics::mae(gTrue, gPredCART),  metrics::mae(gTrue, gPredRF));
    printf("Acuracia (classe)   %8.4f  %8.4f\n",
           metrics::accuracy(cTrue, cPredCART), metrics::accuracy(cTrue, cPredRF));
    auto cmCART = metrics::confusionMatrix(cTrue, cPredCART, {1,2,3,4});
    auto cmRF   = metrics::confusionMatrix(cTrue, cPredRF,   {1,2,3,4});
    printf("F1 macro (classe)   %8.4f  %8.4f\n",
           metrics::f1Macro(cmCART), metrics::f1Macro(cmRF));

    // =================================================================
    // PREDICOES.CSV
    //   - sem arquivo cego: gera 2 CSVs SOBRE O CONJUNTO DE TESTE (450 linhas)
    //     -> previsoes honestas, sem vazamento
    //   - com arquivo cego: gera predicoes_rf.csv usando RF (modelo principal)
    // =================================================================
    printf("\n===========================================\n");
    printf(" GERANDO ARQUIVOS DE PREDICAO\n");
    printf("===========================================\n");

    if (blindPath.empty()) {
        // Sem teste cego: usar split de teste interno (predicoes honestas)
        printf("Sem arquivo cego informado.\n");
        printf("Gerando predicoes sobre o conjunto de TESTE interno (%zu amostras):\n",
               test.nrows());
        writePredictions("predicoes_cart.csv", cartReg, cartClf, test);
        writePredictions("predicoes_rf.csv",   rfReg,   rfClf,   test);
        printf("\nObs: estes arquivos refletem o desempenho real no holdout.\n");
        printf("     Quando receber o arquivo cego do professor, rode:\n");
        printf("       ./main %s <arquivo_cego.txt>\n", trainPath.c_str());
    } else {
        Dataset blind;
        if (!loadBlind(blindPath, blind)) {
            delete cartReg; delete cartClf;
            delete rfReg; delete rfClf;
            return 1;
        }
        printf("Arquivo cego carregado: %zu amostras\n", blind.nrows());
        printf("Gerando predicoes para teste cego:\n");
        writePredictions("predicoes_cart.csv", cartReg, cartClf, blind);
        writePredictions("predicoes_rf.csv",   rfReg,   rfClf,   blind);
        // alias para o nome exigido no enunciado (usa RF como modelo principal)
        writePredictions("predicoes.csv",      rfReg,   rfClf,   blind);
    }

    delete cartReg; delete cartClf;
    delete rfReg;   delete rfClf;
    return 0;
}
