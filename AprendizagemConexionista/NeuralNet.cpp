#include "NeuralNet.hpp"
#include <iostream>

NeuralNet::NeuralNet()
{
}

void NeuralNet::set_activation(Activation a)
{
    for(size_t i=0;i<layers.size();++i)
        layers[i].set_activation(a);
}

void NeuralNet::set_activation_output_layer(Activation a)
{
    if (number_of_layers <= 0) return;
    layers[number_of_layers-1].set_activation(a);
}

void NeuralNet::create_net(int numberoflayers, vector<int> numberneuronslayer, int numberofinputs)
{
    number_of_layers = numberoflayers;
    number_features = numberofinputs;
    number_outputs = numberneuronslayer.empty() ? 0 : numberneuronslayer.back();
    layers.resize(number_of_layers);
    for(size_t i=0;i<number_of_layers;i++)
    {
        number_of_neurons_layer.push_back(numberneuronslayer[i]);
    }

    for (size_t i=0;i<number_of_layers;i++)
    {
        if(i==0)
            layers[i].initialize(number_of_neurons_layer[i], numberofinputs, INPUT);
        else if(i==number_of_layers-1)
            layers[i].initialize(number_of_neurons_layer[i], number_of_neurons_layer[i-1], OUTPUT);
        else
            layers[i].initialize(number_of_neurons_layer[i], number_of_neurons_layer[i-1], INTERMED);
    }
}

void NeuralNet::set_parameters(float momentum, float learn_rate, float max_err)
{
    this->momentum = momentum;
    this->learning_rate = learn_rate;
    this->max_error = max_err;
}

void NeuralNet::update_layer(int number_of_layer)
{
    if (number_of_layer == 0)
    {
        layers[0].update_weights(learning_rate, momentum, training_pair.first);
    }
    else
    {
        layers[number_of_layer].update_weights(learning_rate, momentum, layers[number_of_layer-1].get_outputs());
    }
}

void NeuralNet::propagate_outputs()
{
    for(size_t i=0;i<number_of_layers;i++)
        propagate_outputs(i);
}

void NeuralNet::propagate_outputs(int number_of_layer)
{
    if (number_of_layer == 0)
    {
        layers[0].propagate_outputs(training_pair.first);
    }
    else
    {
        layers[number_of_layer].propagate_outputs(layers[number_of_layer-1].get_outputs());
    }
}

float NeuralNet::propagate_errors(int number_of_layer)
{
    if(number_of_layer == number_of_layers-1)
    {
        return layers[number_of_layer].calculate_layer_error(training_pair.second);
    }
    else
    {
        std::vector<float> erro_next_layer = layers[number_of_layer+1].backpropagation();
        return layers[number_of_layer].calculate_layer_error(erro_next_layer);
    }
}

float NeuralNet::propagate_errors()
{
    float sample_output_err = 0.0f;
    for (int i = number_of_layers - 1; i >= 0; --i)
    {
        float layer_err = propagate_errors(i);
        if (i == number_of_layers - 1)
            sample_output_err = layer_err;
    }
    sum_sq_err += sample_output_err; // accumulate across samples for the epoch (supports batch)
    return sample_output_err;
}

int NeuralNet::predictClassification(const std::vector<double> &features)
{
    if (number_of_layers <= 0)
        return -1;

    std::vector<float> inputs((size_t)number_features, 0.0f);
    size_t limit = features.size() < inputs.size() ? features.size() : inputs.size();
    for (size_t i = 0; i < limit; ++i)
        inputs[i] = static_cast<float>(features[i]);

    layers[0].propagate_outputs(inputs);
    for (int i = 1; i < number_of_layers; ++i)
        layers[i].propagate_outputs(layers[i - 1].get_outputs());

    const std::vector<float> &outputs = layers[number_of_layers - 1].get_outputs();
    if (outputs.empty())
        return -1;

    int best_index = 0;
    float best_value = outputs[0];
    for (size_t i = 1; i < outputs.size(); ++i)
    {
        if (outputs[i] > best_value)
        {
            best_value = outputs[i];
            best_index = static_cast<int>(i);
        }
    }
    return best_index;
}

double NeuralNet::predictRegression(const std::vector<double> &features)
{
    if (number_of_layers <= 0)
        return 0.0;

    std::vector<float> inputs((size_t)number_features, 0.0f);
    size_t limit = features.size() < inputs.size() ? features.size() : inputs.size();
    for (size_t i = 0; i < limit; ++i)
        inputs[i] = static_cast<float>(features[i]);

    layers[0].propagate_outputs(inputs);
    for (int i = 1; i < number_of_layers; ++i)
        layers[i].propagate_outputs(layers[i - 1].get_outputs());

    const std::vector<float> &outputs = layers[number_of_layers - 1].get_outputs();
    if (outputs.empty())
        return 0.0;

    return static_cast<double>(outputs[0]);
}
NeuralNet::~NeuralNet()
{
}
