#ifndef VANILLA_DNN_MODEL_HPP
#define VANILLA_DNN_MODEL_HPP


#include<any>
#include<functional>
#include<vector>
#include"Math/Matrix/Matrix.hpp"
#include"Math/Vector/Vector.hpp"
#include"DNN/Layer/Layer.hpp"
#include "MNIST/MNIST.hpp"

class Model{
	private:
		using  Loss = std::function<float(Vector<float>,Vector<float>)>;
		using  Loss_diff = std::function<Vector<float>(Vector<float>,Vector<float>)>;
		Loss loss;
		Loss_diff loss_diff;
		std::vector<Layer*> Layers; // exclude input&output Layer;
		
		Vector<float> input;
		Vector<float> output;
		
		std::vector<Vector<float>> input_set;
		std::vector<Vector<float>> target_set;
	
		Layer* inputLayer = nullptr;
		Layer* outputLayer = nullptr;
		int depth;
		int nInput,nOutput;
		int batch;
		int epoch;
	
	public:
		Model();
		Model(int _nInput,int _nOutput);
		virtual ~Model();
		void setLoss(Loss _loss);
		void init();
		void learn(int batch,int _epoch);
		void feed_forward();
		void back_propagation();
		void addLayer(Layer* _layer);
		void addLayers(std::vector<Layer*>& _layers);
		void setInput(std::vector<Vector<float>>& _input_set);
		void setOutput(std::vector<Vector<float>>& _target_set);
		
	
	
	
};

#endif
