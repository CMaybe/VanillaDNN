#ifndef VANILLA_DNN_MODEL_HPP
#define VANILLA_DNN_MODEL_HPP

#include<functional>
#include"Math/Matrix/Matrix.hpp"
#include"Math/Vector/Vector.hpp"
#include"DNN/Layer/Layer.hpp"


class Model{
	private:
		using  Loss = std::function<float(Vector<float>,Vector<float>)>;
		Loss loss;
		std::vector<Layer*> Layers; // exclude input&output Layer; 
		Layer* inputLayer = nullptr;
		Layer* outputLayer = nullptr;
		int depth;
	
	public:
		Model();
		virtual ~Model();
		void setLoss(Loss _loss);
		void init();
		void learn();
		void feed_forward();
		void addLayer(Layer* _layer);
		void addLayers(std::vector<Layer*>& _layers);
		
	
	
	
};

#endif
