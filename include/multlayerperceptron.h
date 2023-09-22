#ifndef MULTLAYERPERCEPTRON_H
#define MULTLAYERPERCEPTRON_H

#include "perceptron.h"

class MultiLayerPerceptron{
public:
	std::vector<int> layers;
	double bias;
	double eta;
	std::vector<std::vector<Perceptron>> network;
	std::vector<std::vector<double>> values;
	std::vector<std::vector<double>> d;
	
    MultiLayerPerceptron(std::vector<int> layers,double bias=1.0,double eta=0.5);
	void set_weights(std::vector<std::vector<std::vector<double>>> w_init);
	void print_weights();
	std::vector<double> run(std::vector<double> x);
	double bp(std::vector<double> feature,std::vector<double>label);
};

#endif // MULTLAYERPERCEPTRON_H
