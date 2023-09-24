#include "multlayerperceptron.h"

#include <iostream>
#include <random>
#include <bitset>
#include <algorithm>
using namespace std;


static random_device r_device;
static uniform_real_distribution<double> rnd{0,1};

template<typename T>
string vec2str(const vector<T>& v,string join=","){
	if(v.size()==0){
		return "";
	}
	
	string s=to_string(v[0]);
	
	for(int i=1;i<v.size();i++){
		s+=join+to_string(v[i]);
	}
	
	return s;
}

vector<double> getSegments(uint8_t digit){
	static vector<vector<double>> displays{
		{1,1,1,1,1,1,0},
		{0,1,1,0,0,0,0},
		{1,1,0,1,1,0,1},
		{1,1,1,1,0,0,1},
		{0,1,1,0,0,1,1},
		{1,0,1,1,0,1,1},
		{1,0,1,1,1,1,1},
		{1,1,1,0,0,0,0},
		{1,1,1,1,1,1,1},
		{1,1,1,1,0,1,1}
	};
	
	return displays[digit];
}

string getDisplayAsString(const vector<double>& displays){
	bitset<7> bits;
	
	for(int i=0;i<7;i++){
		bits[i]=(displays[i]>.5);
	}
	
	string display=" ";
	
	display+=(bits[0]?"_":" ");
	display+=" \n";
	
	display+=(bits[5]?"|":" ");	
	display+=(bits[6]?"_":" ");	
	display+=(bits[1]?"|":" ");
	display+="\n";
	
	display+=(bits[4]?"|":" ");	
	display+=(bits[3]?"_":" ");	
	display+=(bits[2]?"|":" ");
	
	return display;
}

double rand_double(){
	return rnd(r_device);
}

void test7to1(MultiLayerPerceptron& sdrnn){
	cout<<endl<<"Trained weights:"<<endl;
	sdrnn.print_weights();
	
	auto runSegment=[&](const vector<double>& segments){
		cout<<vec2str(segments)<<endl;
		
		const auto display=getDisplayAsString(segments);
		const auto rawResult=sdrnn.run(segments);
		const auto result=int(10*rawResult[0]);
		
		cout<<display<<" = "<<rawResult[0]<<" = "<<result<<endl<<endl;
	};
	
	cout<<"digit recognition:"<<endl;
	for(int i=0;i<10;i++){
		const auto segments=getSegments(i);
		
		runSegment(segments);
	}
	
	vector<double> segments;
	segments.resize(7);
	
	for(int i=0;i<5;i++){
		generate(segments.begin(),segments.end(),rand_double);
		
		runSegment(segments);
	}
}

void test7to10(MultiLayerPerceptron& sdrnn){
	cout<<endl<<"Trained weights:"<<endl;
	sdrnn.print_weights();
	
	static auto resultAsInt=[](const vector<double>& v){
		return distance(v.begin(),max_element(v.begin(),v.end()));
	};
	
	auto runSegment=[&](const vector<double>& segments){
		cout<<vec2str(segments)<<endl;
		
		const auto display=getDisplayAsString(segments);
		const auto rawResult=sdrnn.run(segments);
		const auto result=resultAsInt(rawResult);
		
		cout<<display<<" = "<<result<<endl;
		cout<<"confidence: "<<vec2str(rawResult)<<endl<<endl;
	};
	
	cout<<"digit recognition:"<<endl;
	for(int i=0;i<10;i++){
		const auto segments=getSegments(i);
		
		runSegment(segments);
	}
	
	vector<double> segments;
	segments.resize(7);
	
	for(int i=0;i<5;i++){
		generate(segments.begin(),segments.end(),rand_double);
		
		runSegment(segments);
	}
}

int main(int argc,char** argv){
	const int training_steps=300000;
	const int progress_steps=training_steps/30;
	
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
	for(int i=0;i<training_steps;i++){
		MSE=0;
		MSE+=mlp.bp({0,0},{0});
		MSE+=mlp.bp({0,1},{1});
		MSE+=mlp.bp({1,0},{1});
		MSE+=mlp.bp({1,1},{0});
		MSE/=4;
		
		if(i%progress_steps==0){
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
	
	
	cout<<endl<<"trained 7 to 1 network"<<endl;
	auto sdrnn=MultiLayerPerceptron({7,7,1});
	
	cout<<"training..."<<endl;
	for(int i=0;i<training_steps;i++){
		MSE=0;
		MSE += sdrnn.bp({1,1,1,1,1,1,0}, {0.05}); //0 pattern
		MSE += sdrnn.bp({0,1,1,0,0,0,0}, {0.15}); //1 pattern
		MSE += sdrnn.bp({1,1,0,1,1,0,1}, {0.25}); //2 pattern
		MSE += sdrnn.bp({1,1,1,1,0,0,1}, {0.35}); //3 pattern
		MSE += sdrnn.bp({0,1,1,0,0,1,1}, {0.45}); //4 pattern
		MSE += sdrnn.bp({1,0,1,1,0,1,1}, {0.55}); //5 pattern
		MSE += sdrnn.bp({1,0,1,1,1,1,1}, {0.65}); //6 pattern
		MSE += sdrnn.bp({1,1,1,0,0,0,0}, {0.75}); //7 pattern
		MSE += sdrnn.bp({1,1,1,1,1,1,1}, {0.85}); //8 pattern
		MSE += sdrnn.bp({1,1,1,1,0,1,1}, {0.95}); //9 pattern
		MSE/=10;
		
		if(i%progress_steps==0){
			cout<<"i: "<<i<<", MSE: "<<MSE<<endl;
		}
	}
	
	test7to1(sdrnn);
	
	
	cout<<endl<<"trained 7 to 10 network"<<endl;
	sdrnn=MultiLayerPerceptron({7,7,10});
	
	cout<<"training..."<<endl;
	for(int i=0;i<training_steps;i++){
		MSE=0;
		MSE += sdrnn.bp({1,1,1,1,1,1,0}, {1,0,0,0,0,0,0,0,0,0}); //0 pattern
		MSE += sdrnn.bp({0,1,1,0,0,0,0}, {0,1,0,0,0,0,0,0,0,0}); //1 pattern
		MSE += sdrnn.bp({1,1,0,1,1,0,1}, {0,0,1,0,0,0,0,0,0,0}); //2 pattern
		MSE += sdrnn.bp({1,1,1,1,0,0,1}, {0,0,0,1,0,0,0,0,0,0}); //3 pattern
		MSE += sdrnn.bp({0,1,1,0,0,1,1}, {0,0,0,0,1,0,0,0,0,0}); //4 pattern
		MSE += sdrnn.bp({1,0,1,1,0,1,1}, {0,0,0,0,0,1,0,0,0,0}); //5 pattern
		MSE += sdrnn.bp({1,0,1,1,1,1,1}, {0,0,0,0,0,0,1,0,0,0}); //6 pattern
		MSE += sdrnn.bp({1,1,1,0,0,0,0}, {0,0,0,0,0,0,0,1,0,0}); //7 pattern
		MSE += sdrnn.bp({1,1,1,1,1,1,1}, {0,0,0,0,0,0,0,0,1,0}); //8 pattern
		MSE += sdrnn.bp({1,1,1,1,0,1,1}, {0,0,0,0,0,0,0,0,0,1}); //9 pattern
		MSE/=10;
		
		if(i%progress_steps==0){
			cout<<"i: "<<i<<", MSE: "<<MSE<<endl;
		}
	}
	
	test7to10(sdrnn);
	
	return 0;
}
