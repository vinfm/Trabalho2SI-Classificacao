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

void NeuralNetBuilder::train_net(NeuralNet &net, int epochs, unsigned seed)
{
    split_dataset(seed); 

    float current_error = validation_error(net);
    float previous_error = std::numeric_limits<float>::max();
    int epoch = 0;

    while (epoch < epochs && std::fabs(previous_error - current_error) > min_error) 
    {
        net.begin_epoch();
        previous_error = current_error;

        for (int i = 0; i < train_set.nrows(); i++)
        {
            net.set_training_pair({std::vector<float>(train_set.X[i].begin(), train_set.X[i].end()),
                                   std::vector<float>(train_set.Y[i].begin(), train_set.Y[i].end())});
            
            net.propagate_outputs();
            net.propagate_errors();
            net.update_layer_neurons_changes();

            if ((i + 1) % batch_size == 0) {
                for (int j = 0; j < net.get_num_layers(); j++) {
                    net.update_layer(j, batch_size);
                }
            }
        }
        epoch++;
        current_error = validation_error(net);
        std::cout << "Epoch " << epoch << ": Validation Error = " << current_error << std::endl;
        
        split_dataset(seed + epoch);
    }
}

void  NeuralNetBuilder::split_dataset(unsigned seed)
{
    dataset.split(test_fraction, seed, train_set, validation_set);
}

float NeuralNetBuilder::validation_error(NeuralNet &net)
{
    // uses the predict function then compares to validation_set.Y to compute error (e.g. mean squared error for regression, accuracy for classification)

    float error_sum = 0.0f;
    for (size_t i = 0; i < validation_set.nrows(); ++i) {
        std::vector<double> predicted = net.predict(validation_set.X[i]);
        const std::vector<double> &actual = validation_set.Y[i];
        for (size_t j = 0; j < predicted.size() && j < actual.size(); ++j) {
            float err = static_cast<float>(predicted[j] - actual[j]);
            error_sum += err * err; // accumulate squared error
        }
    }
    return error_sum / static_cast<float>(validation_set.nrows()); // return mean squared error
}

NeuralNetBuilder::~NeuralNetBuilder()
{
}
