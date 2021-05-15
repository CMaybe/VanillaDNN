#ifndef VANILLA_DNN_LAYER_HPP
#define VANILLA_DNN_LAYER_HPP

#include<vector>
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
	
		Layer(int _nNueron);
	
	
		Activation activation,activation_diff;
		Layer* preLayer;
		Matrix<float> weight;
		Vector<float> bias;
		Vector<float> inputNeuron; // before activate : preLayer->weight * neuronOutput
		Vector<float> outputNeuron; // after actionte : actinvation(neuronInput)
		
		Matrix<float> dE_dw;
		Vector<float> dE_do;
		Vector<float> do_dz;
		Vector<float> dz_dw;
		Vector<float> dE_dz;
	
		virtual ~Layer();
		void setActivation(Activation _activation);
		int getNueronCnt();
	
	
	
};

#endif
