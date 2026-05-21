#include <vector>
#include <cstdio>

enum LayerType {INPUT, INTERMED, OUTPUT};
class Neuron {
private:
  std::vector<float> weights;
  size_t   number_inputs;
  float out;
  float error;
  std::vector<float> last_change;
  LayerType   kind_of_layer;
public:
  Neuron                 ();
  void initialize(size_t num_dentrites, LayerType kind);
  void update_weights    ( float learn_rate, float momentum,
               const std::vector<float> &outputs_previous_layer );
  float propagation      ( const std::vector<float> &outputs_previous_layer );
  float calculate_error  ( float weighted_error_next_level );
  float back_propagation(size_t index);
  float get_max_output_error(float target);
  float return_output    ( void ) { return(out);};
  ~Neuron                ();
  void print_weights     ( int neuron );
  void save_weights      ( FILE *fileNeuron );
  void load_weights      ( FILE *fileNeuron );
  void reset_error();
};