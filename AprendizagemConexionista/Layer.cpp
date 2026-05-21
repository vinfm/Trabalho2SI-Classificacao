#include "Layer.hpp"
#include <limits>
#include <cmath>

Layer::Layer()
{
}

void Layer::initialize(int num_neurons, int num_inputs_per_neuron, LayerType kind)
{
    number_neurons = num_neurons;
    number_inputs_per_neuron = num_inputs_per_neuron;
    kind_of_layer = kind;
    neurons =  std::vector<Neuron>(number_neurons);
    current_activation = ACT_TANH;
    for(size_t i=0;i<number_neurons;i++)
        neurons[i].initialize(num_inputs_per_neuron, kind);
}

void Layer::update_weights(float learn_rate, float momentum, int batch_size)
{
    for(size_t i=0;i<number_neurons;i++)
    {
        neurons[i].update_weights(learn_rate, momentum, batch_size);
    }
}

void Layer::propagate_outputs(const std::vector<float> &outputs_previous_layer)
{
    outputs.clear();
    if (current_activation == ACT_SOFTMAX)
    {
        std::vector<float> sums;
        sums.reserve(number_neurons);
        float max_sum = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < (size_t)number_neurons; ++i) {
            float s = neurons[i].compute_sum(outputs_previous_layer);
            sums.push_back(s);
            if (s > max_sum) max_sum = s;
        }
        float sum_exp = 0.0f;
        std::vector<float> exps(number_neurons);
        for (size_t i = 0; i < (size_t)number_neurons; ++i) {
            float e = expf(sums[i] - max_sum);
            exps[i] = e;
            sum_exp += e;
        }
        for (size_t i = 0; i < (size_t)number_neurons; ++i) {
            float val = exps[i] / sum_exp;
            neurons[i].set_output_value(val);
            outputs.push_back(val);
        }
    }
    else
    {
        for(size_t i=0;i<number_neurons;i++)
        {
            outputs.push_back(neurons[i].propagation(outputs_previous_layer));
        }
    }
}

float Layer::calculate_layer_error(const std::vector<float> &erro_next_layer)
{   errors.clear();
    for (size_t i=0;i<number_neurons;i++)
    {
        errors.push_back(neurons[i].calculate_error(erro_next_layer[i]));
    }   

    float error_sum = 0.0f;
    for (size_t i=0;i<errors.size();i++)        error_sum += errors[i]*errors[i];
    return error_sum;
}

const std::vector<float> &Layer::get_outputs() const
{
    return outputs;
}

std::vector<float> Layer::backpropagation()
{
    std::vector<float> weighted_error_sum((size_t)number_inputs_per_neuron, 0.0f);
    for (size_t n = 0; n < (size_t)number_neurons; ++n)
    {
        for (size_t k = 0; k < (size_t)number_inputs_per_neuron; ++k)
        {
            weighted_error_sum[k] += neurons[n].back_propagation(k);
        }
    }
    return weighted_error_sum;
}

void Layer::reset_errors()
{
    for(size_t i=0;i<number_neurons;i++)
        neurons[i].reset_error();
    errors.clear();
}

void Layer::set_activation(Activation a)
{
    current_activation = a;
    for(size_t i=0;i<number_neurons;i++)
        neurons[i].set_activation(a);
}

void Layer::update_weights_changes(const std::vector<float> &outputs_previous_layer)
{
    for(size_t i=0;i<number_neurons;i++)
        neurons[i].update_weights_changes(outputs_previous_layer);
}

Layer::~Layer()
{
}
