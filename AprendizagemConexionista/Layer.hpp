#ifndef LAYER_HPP
#define LAYER_HPP
#include "Neuron.hpp"
class Layer {
private:
    int layer_number;
    LayerType kind_of_layer;
    int number_neurons;
    int number_inputs_per_neuron;
    std::vector<Neuron> neurons;
    std::vector<float> outputs;
    std::vector<float> errors;
public:
    Layer();
    void initialize(int num_neurons, int num_inputs_per_neuron, LayerType kind);
    void update_weights(float learning_rate, float momentum, const std::vector<float> &outputs_previous_layer);
    void propagate_outputs(const std::vector<float> &outputs_previous_layer);
    float calculate_layer_error(const std::vector<float> &erro_next_layer);
    const std::vector<float> &get_outputs() const;
    float backpropagation(int index);
    std::vector<float> backpropagation();
    void print_layer(int layer);
    void reset_errors();
    void save_layer(FILE *fileLayer);
    void load_layer(FILE *fileLayer);
    ~Layer();
};
#endif