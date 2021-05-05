#ifndef VANILLA_DNN_LAYER_HPP
#define VANILLA_DNN_LAYER_HPP

#include<functional>
#include"Math/Matrix/Matrix.hpp"
#include"Math/Vector/Vector.hpp"

class Layer{
	private:
		using func = std::function<float(Vector<float>,Vector<float>)> ;
		func activation,loss;
		Layer* preLayer;
		Matrix<float> weight;
		Vector<float> neuronInput; // before activate : preLayer->weight * neuronOutput
		Vector<float> neuronOutput; // after actionte : activatoin(neuronInput)
	
	public:
		Layer();
		virtual ~Layer(){}
		void setActivation(void* f);
		void setLoss(void* f);
	
	
	
};

#endif
