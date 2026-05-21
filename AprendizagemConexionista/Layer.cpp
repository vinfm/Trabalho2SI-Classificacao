#include "Layer.hpp"

Layer::Layer()
{
}

void Layer::initialize(int num_neurons, int num_inputs_per_neuron, LayerType kind)
{
    number_neurons = num_neurons;
    number_inputs_per_neuron = num_inputs_per_neuron;
    kind_of_layer = kind;
    neurons =  std::vector<Neuron>(number_neurons);
    for(size_t i=0;i<number_neurons;i++)
        neurons[i].initialize(num_inputs_per_neuron, kind);
}

void Layer::update_weights(float learn_rate, float momentum, const std::vector<float> &outputs_previous_layer)
{
    for(size_t i=0;i<number_neurons;i++)
    {
        neurons[i].update_weights(learn_rate, momentum, outputs_previous_layer);
    }
}

void Layer::propagate_outputs(const std::vector<float> &outputs_previous_layer)
{
    outputs.clear();
    for(size_t i=0;i<number_neurons;i++)
    {
        outputs.push_back(neurons[i].propagation(outputs_previous_layer));
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

Layer::~Layer()
{
}
