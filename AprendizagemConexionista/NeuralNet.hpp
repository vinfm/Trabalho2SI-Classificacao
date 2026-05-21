#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP
#include "Layer.hpp"
#include <vector>
using namespace std;

class NeuralNet {
protected:
  int number_features;
  int number_outputs;
  long          count;
  float         momentum;
  float         learning_rate;
  float         max_error;
  float         max_error_epoch;
  float		    sum_sq_err;
  int           number_of_layers;
  vector<int> number_of_neurons_layer;
  vector<Layer> layers;
  pair<vector<float>, vector<float>> training_pair;
  vector<float> neuron_outputs;
public:
  NeuralNet             ();
  void create_net        ( int numberoflayers,
		           vector<int> numberneuronslayer,
		           int numberofinputs );
  void set_training_pair   ( pair<vector<float>, vector<float>> tp ) { training_pair = tp;};
  int  get_num_layers    () { return( number_of_layers);};
  void set_parameters    ( float momentum, float learn_rate, float max_err );
  void update_layer      ( int number_of_layer );
  void propagate_outputs();
  void propagate_outputs(int number_of_layer);
  float propagate_errors ( int number_of_layer );
  float propagate_errors();
  int predictClassification(const std::vector<double> &features);
  void set_activation(Activation a);
  void set_activation_output_layer(Activation a);
  double predictRegression(const std::vector<double> &features);
  ~NeuralNet();
  void print_net         ( int layer );
  void save_net          (char *extension);
  void load_net          (char *extension);
  void begin_epoch	 (){ 
      sum_sq_err = 0.0f;
      for (size_t i = 0; i < layers.size(); ++i)
          layers[i].reset_errors();
  };
  float get_sum_sq_err   (){return(sum_sq_err);};

};

#endif
