#ifndef VANILLA_DNN_NEURON_HPP
#define VANILLA_DNN_NEURON_HPP

#include<vector>

class Neuron{
	public:
		Neuron();
		virtual ~Node();
	private:
		vector<float> input,weight; // weight : pre -> cur;
		float output;
};

#endif