#ifndef VANILLA_DNN_LAYER_HPP
#define VANILLA_DNN_LAYER_HPP

#include<functional>
#include"Math/Matrix/Matrix.hpp"
#include"Math/Vector/Vector.hpp"


class Layer{
	private:
		using Activation = std::function<Vector<float>(Vector<float>)>;
		Activation activation;
		Layer* preLayer;
		Matrix<float> weight;
		Vector<float> neuronInput; // before activate : preLayer->weight * neuronOutput
		Vector<float> neuronOutput; // after actionte : actinvation(neuronInput)
	
	public:
		Layer();
		virtual ~Layer(){}
		void setActivation(Activation _activation);
	
	
	
};

#endif
