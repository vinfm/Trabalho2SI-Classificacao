#ifndef NEURAL_NET_BUILDER_HPP
#define NEURAL_NET_BUILDER_HPP
#include "NeuralNet.hpp"
#include "Dataset.hpp"
#include <vector>
#include <string>
using namespace std;

class NeuralNetBuilder {
private:
    Dataset dataset;
    Dataset train_set;
    Dataset validation_set;
    float min_error;
    int batch_size;
public:
    NeuralNetBuilder();
    NeuralNetBuilder(const Dataset& ds) : dataset(ds) {}
    NeuralNet build_net(int numberoflayers, vector<int> numberneuronslayer, int numberofinputs);
    void set_parameters(NeuralNet& net, float momentum, float learn_rate, float max_err, int batch_size);
    void train_net(NeuralNet& net, int epochs, float test_fraction, unsigned seed);
    ~NeuralNetBuilder();
};
#endif