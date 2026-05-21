#include "Neuron.hpp"
#include "math.h"
#include <iostream>
#include <cstring>
#include <random>
using namespace std;

static std::mt19937 &get_global_rng()
{
    static thread_local std::mt19937 rng((std::random_device())());
    return rng;
}

Neuron::Neuron()
{
  out = 0;
  error = 0;
}

void Neuron::initialize (size_t num_dentrites, LayerType kind )
{
  size_t i;

  number_inputs = num_dentrites;
  kind_of_layer = kind;
  weights.resize(num_dentrites+1);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for(i=0;i<num_dentrites;i++)
    weights[i] = dist(get_global_rng());
  weights[num_dentrites] = dist(get_global_rng());
  last_change.resize(num_dentrites+1);
  for(i=0;i<=num_dentrites;i++)
    last_change[i] = 0.0f;
  last_sum = 0.0f;

}

float Neuron::get_max_output_error(float target)
{
int DEBUG=0;
float err;

 err = target - out;
 if(err<0) err = -1*err;

if(DEBUG)
{
  printf("target = %f\t", target);
  printf("out = %f\t", out);
  printf("error = %f\n", err);
}
return(err);
}

float Neuron::calculate_error ( float weighted_error_next_layer )
{
  float target;
  switch(kind_of_layer)
    {
    case OUTPUT:
      target = weighted_error_next_layer;
      if (activation == ACT_SOFTMAX) {
        // softmax + cross-entropy: gradient simplifies to (target - out)
        error += (target - out);
      } else if (activation_derivative) {
        error += activation_derivative(out) * (target - out);
      } else {
        error += (1.0f - out*out) * (target - out);
      }
      break;
    default:
      if (activation_derivative)
        error += activation_derivative(out) * weighted_error_next_layer;
      else
        error += (1.0f - out*out) * weighted_error_next_layer;
      break;
    }
  return(error);
}

float Neuron::back_propagation ( size_t index )
{
  float weighted_error;
  weighted_error = error * weights[index];
  return(weighted_error);
}

void Neuron::update_weights(float learn_rate, float momentum, const std::vector<float> &outputs_previous_layer)
{
    float change;
    for(size_t i=0;i<number_inputs;i++)
    {
      change = learn_rate * error * outputs_previous_layer[i] + momentum * last_change[i];
      weights[i] += change;
      last_change[i] = change;
    }
    change = learn_rate * error + momentum * last_change[number_inputs];
    weights[number_inputs] += change;
    last_change[number_inputs] = change;
    error = 0.0f;
}

float Neuron::propagation(const std::vector<float> &outputs_previous_layer)
{
  float sum = 0.0f;
  for(size_t i=0;i<number_inputs;i++)
    sum += weights[i] * outputs_previous_layer[i];
  sum += weights[number_inputs]; // Bias term
  last_sum = sum;
  if (activation == ACT_SOFTMAX) {
    if (activation_function)
      out = activation_function(sum); // exp(sum)
    else
      out = expf(sum);
  } else {
    out = activation_function(sum);
  }
  return out;
}

float Neuron::compute_sum(const std::vector<float> &outputs_previous_layer)
{
    float sum = 0.0f;
    for(size_t i=0;i<number_inputs;i++)
      sum += weights[i] * outputs_previous_layer[i];
    sum += weights[number_inputs];
    last_sum = sum;
    return sum;
}

void Neuron::set_output_value(float v)
{
    out = v;
}

Neuron::~Neuron ()
{
}

void Neuron::print_weights ( int neuron )
{
  cout << "   Neuron " << neuron << "\n";
  cout << "   Weights :\n";
  for(size_t i=0;i<number_inputs;i++)
    cout << "\t" << i << "  " << weights[i] << "\n";
}

void Neuron::save_weights ( FILE *fileNeuron )
{
  fprintf (fileNeuron, "%d\n", (int)number_inputs);
  for (size_t i=0;i<number_inputs;i++)
      fprintf (fileNeuron, "%f\n", weights[i]);
}

void Neuron::load_weights ( FILE *fileNeuron )
{
  char line[30];

  if(fgets(line, 30, fileNeuron) == NULL)
      cout << "Error!";
  line[strlen(line)-1] = '\0';
  int ni = atoi (line);
  number_inputs = (size_t) ni;
  weights.resize(number_inputs+1);
  for (int i=0;i<ni;i++)
    {
      if(fgets(line, 30, fileNeuron) == NULL)
          cout << "Error!";
      line[strlen(line)-1] = '\0';
      weights[i] = (float) atof (line);
    }
}

void Neuron::reset_error()
{
    error = 0.0f;
}

void Neuron::set_activation(Activation a)
{
  activation = a;
  switch (activation)
  {
    case ACT_TANH:
      activation_function = tanh_activation;
      activation_derivative = tanh_derivative;
      break;
    case ACT_SIGMOID:
      activation_function = sigmoid_activation;
      activation_derivative = sigmoid_derivative;
      break;
    case ACT_SOFTMAX:
      activation_function = softmax_exp;
      activation_derivative = nullptr;
      break;
    default:
      activation_function = tanh_activation;
      activation_derivative = tanh_derivative;
      break;
  }
}
