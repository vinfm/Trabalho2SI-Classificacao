#ifndef NEURON_HPP
#define NEURON_HPP
#include <vector>
#include <cstdio>
#include <cmath>

enum LayerType {INPUT, INTERMED, OUTPUT};
enum Activation {ACT_TANH, ACT_SIGMOID, RELU, ACT_SOFTMAX, ACT_LINEAR};
class Neuron {
private:
  std::vector<float> weights;
  size_t   number_inputs;
  float out;
  float last_sum;
  float error;
  std::vector<float> last_change;
  std::vector<float> current_change;
  LayerType   kind_of_layer;
  Activation activation;
public:
  Neuron                 ();
  void initialize(size_t num_dentrites, LayerType kind);
  float propagation      ( const std::vector<float> &outputs_previous_layer );
  float compute_sum      ( const std::vector<float> &outputs_previous_layer );
  void set_output_value  ( float v );
  float calculate_error  ( float weighted_error_next_level );
  float back_propagation(size_t index);
  void update_weights(float learning_rate, float momentum, int batch_size);
  void update_weights_changes(const std::vector<float> &outputs_previous_layer);
  float get_max_output_error(float target);
  float return_output    ( void ) { return(out);};
  ~Neuron                ();
  void print_weights     ( int neuron );
  void save_weights      ( FILE *fileNeuron );
  void load_weights      ( FILE *fileNeuron );
  void reset_error();
  void set_activation(Activation a);
  void normalize_change(int batch_size);
  float (*activation_function)(float);
  float (*activation_derivative)(float);
  static float tanh_activation(float x) { return std::tanh(x); }
  static float sigmoid_activation(float x) { return 1.0f / (1.0f + std::exp(-x)); }
  static float relu_activation(float x) { return x > 0.0f ? x : 0.0f; }
  static float linear_activation(float x) { return x; }
  static float softmax_exp(float x) { return std::exp(x); }
  static float tanh_derivative(float y) { return 1.0f - y*y; } // input y is activation output
  static float sigmoid_derivative(float y) { return y * (1.0f - y); }
  static float relu_derivative(float y) { return y > 0 ? 1.0f : 0.0f; }
  static float linear_derivative(float) { return 1.0f; }
};

#endif