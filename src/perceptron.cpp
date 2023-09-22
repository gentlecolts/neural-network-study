#include "perceptron.h"
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>

static std::random_device r_device;
static std::uniform_real_distribution<double> rnd{0,1};

static double frand(){
	return 2.0*rnd(r_device)-1.0;
}

Perceptron::Perceptron(int inputs,double bias){
	this->bias=bias;
	
	weights.resize(inputs+1);
	std::generate(weights.begin(),weights.end(),frand);
}

double Perceptron::run(std::vector<double> x){
	x.push_back(bias);
	const auto sum=std::inner_product(x.begin(),x.end(),weights.begin(),0.0);
	return sigmoid(sum);
}

void Perceptron::set_weights(std::vector<double> w_init){
	weights=w_init;
}

double Perceptron::sigmoid(double x){
	return 1/(1+std::exp(-x));
}
