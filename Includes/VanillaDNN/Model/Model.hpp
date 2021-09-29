#ifndef VANILLA_DNN_MODEL_HPP
#define VANILLA_DNN_MODEL_HPP


#include <any>
#include <functional>
#include <vector>
#include <iostream>
#include <random>
#include <future>
#include <thread>
#include <VanillaDNN/Math/Matrix/Matrix.hpp>
#include <VanillaDNN/Math/Vector/Vector.hpp>
#include <VanillaDNN/Layers/Layer.hpp>
#include <VanillaDNN/Layers/DenseLayer.hpp>
#include <VanillaDNN/Functions/Optimizer.hpp>
#include <VanillaDNN/MNIST/MNIST.hpp>

class Model {
private:
	using Activation = std::function<Vector<float>(Vector<float>&)>;
	using  Loss = std::function<float(Vector<float>&, Vector<float>&)>;
	using  Loss_diff = std::function<Vector<float>(Vector<float>&, Vector<float>&)>;
	Loss loss;
	Loss_diff loss_diff;
	
	Optimizer optimizer;
	std::vector<Layer*> layers; // exclude input&output Layer;
	
	Vector<float> input;
	Vector<float> target;
	Vector<float> output;

	std::vector<Vector<float>> input_set;
	std::vector<Vector<float>> target_set;
	std::vector<Vector<float>> output_set;
	
	std::vector<Matrix<float>> batch_dE_dw;
	std::vector<Vector<float>> batch_dE_db;

	Layer* inputLayer = nullptr;
	Layer * outputLayer = nullptr;
	
	int batch_size;
	int epoch;
	int nEval;
	int total;
	float accuracy;
	float error;

public:
	Model();
	virtual ~Model();
	void setLoss(Loss _loss);
	
	void init();
	void fit(int _total, int _epoch, int _batch = 1);
	
	void evaluate(int _batch, bool show = false);
	void feed_forward(int idx);
	void predict(int idx);
	void back_propagation(int idx);
	void update();
	
	void addLayer(Layer* _layer);
	
	void setInput(std::vector<Vector<float>>& _input_set);
	void setTarget(std::vector<Vector<float>>& _target_set);
	
	void setOptimizer(Optimizer _optimizer);
	
	float getAccuracy();
	float getError();
	void setLearningRate(float lr);



};

#endif
