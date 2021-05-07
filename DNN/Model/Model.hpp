#ifndef VANILLA_DNN_MODEL_HPP
#define VANILLA_DNN_MODEL_HPP

#include<functional>
#include"Math/Matrix/Matrix.hpp"
#include"Math/Vector/Vector.hpp"
#include"DNN/Layer/Layer.hpp"


class Model{
	private:
		typedef std::function<float(Vector<float>,Vector<float>)> Loss;
		Loss loss;
		std::vector<Layer*> Layers; // exclude input&output Layer; 
		Layer* inputLayer = nullptr;
		Layer* outputLayer = nullptr;
	
	public:
		Model();
		virtual ~Model(){}
		void setLoss(Loss _loss);
	
	
	
};

#endif
