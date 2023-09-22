#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>


class Perceptron{
public:
	double bias;
	std::vector<double> weights;
	
    Perceptron(int inputs,double bias=1.0);
	double run(std::vector<double> x);
	void set_weights(std::vector<double> w_init);
	double sigmoid(double x);
};

#endif // PERCEPTRON_H
