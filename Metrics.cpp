#include "Metrics.hpp"

#include <cmath>
#include <cstdio>
#include <string>

namespace metrics {

double rmse(const std::vector<double>& yTrue, const std::vector<double>& yPred) {
    if (yTrue.empty() || yTrue.size() != yPred.size()) return 0.0;
    double s = 0.0;
    for (std::size_t i = 0; i < yTrue.size(); ++i) {
        double delta = yTrue[i] - yPred[i];
        s += delta * delta;
    }
    return std::sqrt(s / static_cast<double>(yTrue.size()));
}

double mae(const std::vector<double>& yTrue, const std::vector<double>& yPred) {
    if (yTrue.empty() || yTrue.size() != yPred.size()) return 0.0;
    double s = 0.0;
    for (std::size_t i = 0; i < yTrue.size(); ++i) s += std::fabs(yTrue[i] - yPred[i]);
    return s / static_cast<double>(yTrue.size());
}

double accuracy(const std::vector<int>& yTrue, const std::vector<int>& yPred) {
    if (yTrue.empty() || yTrue.size() != yPred.size()) return 0.0;
    int hits = 0;
    for (std::size_t i = 0; i < yTrue.size(); ++i) if (yTrue[i] == yPred[i]) ++hits;
    return static_cast<double>(hits) / static_cast<double>(yTrue.size());
}

std::vector<std::vector<int>> confusionMatrix(const std::vector<int>& yTrue, const std::vector<int>& yPred, const std::vector<int>& classes) {
    std::size_t k = classes.size();
    std::vector<std::vector<int>> cm(k, std::vector<int>(k, 0));
    auto idx = [&](int c) -> int {
        for (std::size_t i = 0; i < classes.size(); ++i) if (classes[i] == c) return static_cast<int>(i);
        return -1;
    };
    for (std::size_t i = 0; i < yTrue.size(); ++i) {
        int a = idx(yTrue[i]);
        int b = idx(yPred[i]);
        if (a >= 0 && b >= 0) cm[a][b]++;
    }
    return cm;
}

double precisionMacro(const std::vector<std::vector<int>>& cm) {
    std::size_t k = cm.size();
    if (k == 0) return 0.0;
    double sum = 0.0;
    for (std::size_t j = 0; j < k; ++j) {
        int tp = cm[j][j];
        int colSum = 0;
        for (std::size_t i = 0; i < k; ++i) colSum += cm[i][j];
        sum += (colSum > 0) ? static_cast<double>(tp) / colSum : 0.0;
    }
    return sum / static_cast<double>(k);
}

double recallMacro(const std::vector<std::vector<int>>& cm) {
    std::size_t tam = cm.size();
    if (tam == 0) return 0.0;
    double soma = 0.0;
    for (std::size_t i = 0; i < tam; ++i) {
        int tp = cm[i][i];
        int somaLinha = 0;
        for (std::size_t j = 0; j < tam; ++j) somaLinha += cm[i][j];
        soma += (somaLinha > 0) ? static_cast<double>(tp) / somaLinha : 0.0;
    }
    return soma / static_cast<double>(tam);
}

double f1Macro(const std::vector<std::vector<int>>& cm) {
    std::size_t k = cm.size();
    if (k == 0) return 0.0;
    double sum = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        int tp = cm[i][i];
        int rowSum = 0, colSum = 0;
        for (std::size_t j = 0; j < k; ++j) { rowSum += cm[i][j]; colSum += cm[j][i]; }
        double prec = (colSum > 0) ? static_cast<double>(tp) / colSum : 0.0;
        double rec  = (rowSum > 0) ? static_cast<double>(tp) / rowSum : 0.0;
        sum += (prec + rec > 0.0) ? (2.0 * prec * rec) / (prec + rec) : 0.0;
    }
    return sum / static_cast<double>(k);
}

void printRegressionReport(const std::vector<double>& yTrue, const std::vector<double>& yPred, const std::string& title) {
    printf("\n=== Resultado: %s ===\n", title.c_str());
    printf("Erro médio (RMSE): %.4f\n", rmse(yTrue, yPred));
    printf("Erro absoluto médio (MAE): %.4f\n", mae(yTrue, yPred));
}

void printClassificationReport(const std::vector<int>& yTrue, const std::vector<int>& yPred, const std::vector<int>& classes, const std::string& title) {
    auto cm = confusionMatrix(yTrue, yPred, classes);
    printf("\n=== Resultado: %s ===\n", title.c_str());
    printf("Acurácia: %.4f\n", accuracy(yTrue, yPred));
    printf("Precisão média: %.4f\n", precisionMacro(cm));
    printf("Revocação média: %.4f\n", recallMacro(cm));
    printf("F1 média: %.4f\n", f1Macro(cm));

    printf("\nMatriz de confusão (linha=real, coluna=previsto):\n");
    printf("        ");
    for (int c : classes) printf("%6d", c);
    printf("\n");
    for (std::size_t i = 0; i < classes.size(); ++i) {
        printf("%6d |", classes[i]);
        for (std::size_t j = 0; j < classes.size(); ++j) printf("%6d", cm[i][j]);
        printf("\n");
    }
}

}