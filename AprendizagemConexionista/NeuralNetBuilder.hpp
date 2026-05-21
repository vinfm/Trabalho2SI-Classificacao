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
    float test_fraction;
public:
    NeuralNetBuilder();
    NeuralNetBuilder(const Dataset& ds) : dataset(ds) {}
    NeuralNet build_net(int numberoflayers, vector<int> numberneuronslayer, int numberofinputs);
    void set_parameters(NeuralNet& net, float momentum, float learn_rate, float max_err, int batch_size, float test_fraction, float min_error) {
        net.set_parameters(momentum, learn_rate, max_err);
        this->batch_size = batch_size;
        this->test_fraction = test_fraction;
        this->min_error = min_error;
    }
    void train_net(NeuralNet& net, int epochs, unsigned seed);
    void split_dataset(unsigned seed);
    float validation_error(NeuralNet &net);
    ~NeuralNetBuilder();
};
#endif