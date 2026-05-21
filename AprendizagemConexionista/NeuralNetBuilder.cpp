#include "NeuralNetBuilder.hpp"
#include <iostream>
#include <limits>

NeuralNetBuilder::NeuralNetBuilder()
{
}

NeuralNet NeuralNetBuilder::build_net(int numberoflayers, vector<int> numberneuronslayer, int numberofinputs)
{
    NeuralNet net;
    net.create_net(numberoflayers, numberneuronslayer, numberofinputs);
    return net;
}

void NeuralNetBuilder::set_parameters(NeuralNet &net, float momentum, float learn_rate, float max_err, int batch_size)
{
    net.set_parameters(momentum, learn_rate, max_err);
    this->batch_size = batch_size;
}

void NeuralNetBuilder::train_net(NeuralNet &net, int epochs, float test_fraction, unsigned seed)
{
    float previous_error = std::numeric_limits<float>::max();
    int epoch = 0;
    while (epoch < epochs && abs(previous_error - net.get_sum_sq_err()) > min_error) 
    {
        previous_error = net.get_sum_sq_err();
        net.begin_epoch();
        for (int i=0; i<train_set.nrows(); i++)
        {
            net.set_training_pair({std::vector<float>(train_set.X[i].begin(), train_set.X[i].end()),
                               std::vector<float>(train_set.Y[i].begin(), train_set.Y[i].end())});
            net.propagate_outputs();

            net.propagate_errors();
            
            if ((i+1) % batch_size == 0 || i == train_set.nrows() - 1) {
                for (int j=0; j<net.get_num_layers(); j++) {
                    net.update_layer(j);
                }
            }
        }

        epoch++;
        std::cout << "Epoch " << epoch << ": Sum of Squared Errors = " << net.get_sum_sq_err() << std::endl;
    }
}
NeuralNetBuilder::~NeuralNetBuilder()
{
}
