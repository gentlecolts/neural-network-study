#include "multlayerperceptron.h"

#include <iostream>
using namespace std;

int main(int argc,char** argv){
	
	auto p=new Perceptron(2);
	
	//p->set_weights({10,10,-15});//AND gate
	//p->set_weights({10,10,-5});//OR gate
	p->set_weights({-10,-10,15});//NAND gate
	
	cout<<"Gate: "<<endl;
	cout<<p->run({0,0})<<endl;
	cout<<p->run({0,1})<<endl;
	cout<<p->run({1,0})<<endl;
	cout<<p->run({1,1})<<endl;
	
	cout<<"multi-layer"<<endl;
	MultiLayerPerceptron mlp({2,2,1});
	mlp.set_weights({
		{{-10,-10,15},{15,15,-10}},
		{{10,10,-15}}
	});
	cout<<"weights:"<<endl;
	mlp.print_weights();
	
	cout<<"XOR:"<<endl;
	cout<<"0 0 ="<<mlp.run({0,0})[0]<<endl;
	cout<<"0 1 ="<<mlp.run({0,1})[0]<<endl;
	cout<<"1 0 ="<<mlp.run({1,0})[0]<<endl;
	cout<<"1 1 ="<<mlp.run({1,1})[0]<<endl;
	
	
	cout<<endl<<"trained xor"<<endl;
	mlp=MultiLayerPerceptron({2,2,1});
	
	cout<<"training..."<<endl;
	double MSE;
	for(int i=0;i<3000;i++){
		MSE=0;
		MSE+=mlp.bp({0,0},{0});
		MSE+=mlp.bp({0,1},{1});
		MSE+=mlp.bp({1,0},{1});
		MSE+=mlp.bp({1,1},{0});
		MSE/=4;
		
		if(i%100==0){
			cout<<"i: "<<i<<", MSE: "<<MSE<<endl;
		}
	}
	
	cout<<endl<<"Trained weights:"<<endl;
	mlp.print_weights();
	
	cout<<"XOR:"<<endl;
	cout<<"0 0 ="<<mlp.run({0,0})[0]<<endl;
	cout<<"0 1 ="<<mlp.run({0,1})[0]<<endl;
	cout<<"1 0 ="<<mlp.run({1,0})[0]<<endl;
	cout<<"1 1 ="<<mlp.run({1,1})[0]<<endl;
	return 0;
}
