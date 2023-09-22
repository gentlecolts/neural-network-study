#include "multlayerperceptron.h"

#include <iostream>

MultiLayerPerceptron::MultiLayerPerceptron(std::vector<int> layers,double bias,double eta){
	this->layers=layers;
	this->bias=bias;
	this->eta=eta;
	
	for(int i=0;i<layers.size();i++){
		values.push_back(std::vector<double>(layers[i],0.0));
		d.push_back(std::vector<double>(layers[i],0.0));
		network.push_back(std::vector<Perceptron>());
		
		if(i>0){
			for(int j=0;j<layers[i];j++){
				network[i].push_back(Perceptron(layers[i-1],bias));
			}
		}
	}
}

void MultiLayerPerceptron::set_weights(std::vector<std::vector<std::vector<double>>> w_init){
	for(int layer=0;layer<w_init.size();layer++){
		for(int i=0;i<w_init[layer].size();i++){
			//w_init has one less layer, as it does not include the input layer
			network[layer+1][i].set_weights(w_init[layer][i]);
		}
	}
}

void MultiLayerPerceptron::print_weights(){
	std::cout<<std::endl;
	
	for(int i=1;i<network.size();i++){
		for(int j=0;j<layers[i];j++){
			std::cout<<"Layer"<<i+1<<" Neuron "<<j<<": ";
			
			for(const auto& it:network[i][j].weights){
				std::cout<<it<<"   ";
			}
			std::cout<<std::endl;
		}
	}
	std::cout<<std::endl;
}

std::vector<double> MultiLayerPerceptron::run(std::vector<double> x){
	values[0]=x;
	for(int i=1;i<network.size();i++){
		for(int j=0;j<layers[i];j++){
			values[i][j]=network[i][j].run(values[i-1]);
		}
	}
	return values.back();
}

double MultiLayerPerceptron::bp(std::vector<double> feature, std::vector<double> label){
	//std::cout<<"feed sample to network"<<std::endl;
	auto outputs=run(feature);
	
	//std::cout<<"calculate mean square error"<<std::endl;
	std::vector<double> error;
	double mse=0;
	for(int i=0;i<label.size();i++){
		error.push_back(label[i]-outputs[i]);
		mse+=error[i]*error[i];
	}
	mse/=layers.back();
	
	//std::cout<<"calculate output error terms"<<std::endl;
	for(int i=0;i<outputs.size();i++){
		//std::cout<<i<<std::endl;
		d.back()[i]=outputs[i]*(1-outputs[i])*(error[i]);
	}
	
	//std::cout<<"calculate error term of each unit on each layer"<<std::endl;
	for(int i=network.size()-2;i>0;i--){
		for(int h=0;h<network[i].size();h++){
			double fwd_error=0;
			for(int k=0;k<layers[i+1];k++){
				fwd_error+=network[i+1][k].weights[h]*d[i+1][k];
			}
			d[i][h]=values[i][h]*(1-values[i][h])*fwd_error;
		}
	}
	
	//std::cout<<"calculate deltas, update weights"<<std::endl;
	for(int i=1;i<network.size();i++){//layers
		for(int j=0;j<layers[i];j++){//neurons
			for(int k=0;k<layers[i-1]+1;k++){//inputs
				double delta;
				if(k==layers[i-1]){
					delta=eta*d[i][j]*bias;
				}else{
					delta=eta*d[i][j]*values[i-1][k];
				}
				network[i][j].weights[k]+=delta;
			}
		}
	}
	
	return mse;
}
