#ifndef VANILLA_DNN_LAYER_HPP
#define VANILLA_DNN_LAYER_HPP

#include<functional>
#include<memory>
#include"Math/Matrix/Matrix.hpp"
#include"Math/Vector/Vector.hpp"
#include"DNN/Functions/DNNFunction.hpp"


class Layer{
	private:
		using Activation = std::function<Vector<float>(Vector<float>)>;
		int nNueron;
	
	public:
		Activation activation,activation_diff;
		Layer* preLayer;
		Matrix<float> weight;
		Vector<float> bias;
		Vector<float> inputNeuron; // before activate : preLayer->weight * neuronOutput
		Vector<float> outputNeuron; // after actionte : actinvation(neuronInput)
		Layer(int _nNueron);
		virtual ~Layer();
		void setActivation(Activation _activation);
		int getNueronCnt();
	
	
	
};

#endif
