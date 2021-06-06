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
#include <VanillaDNN/DNN/Layers/Layer.hpp>
#include <VanillaDNN/DNN/Functions/Optimizer.hpp>
#include <VanillaDNN/MNIST/MNIST.hpp>

class Model {
private:
	using Activation = std::function<Vector<float>(Vector<float>)>;
	using  Loss = std::function<float(Vector<float>, Vector<float>)>;
	using  Loss_diff = std::function<Vector<float>(Vector<float>, Vector<float>)>;
	Loss loss;
	Loss_diff loss_diff;
	
	Optimizer *optimizer = nullptr;
	std::vector<Layer*> Layers; // exclude input&output Layer;
	
	Vector<float> input;
	Vector<float> target;
	Vector<float> output;

	std::vector<Vector<float>> input_set;
	std::vector<Vector<float>> target_set;;
	std::vector<Vector<float>> output_set;
	
	std::vector<Matrix<float>> batch_dE_dw;
	std::vector<Vector<float>> batch_dE_do;
	std::vector<Vector<float>> batch_do_dz;
	std::vector<Vector<float>> batch_dz_dw;
	std::vector<Vector<float>> batch_dE_dz;
	std::vector<Vector<float>> batch_dE_db;
	std::vector<Vector<float>> batch_dz_db;

	Layer* inputLayer = nullptr;
	Layer* outputLayer = nullptr;
	int depth;
	int nInput, nOutput;
	int batch_size;
	int epoch;
	int nEval;
	int total;
	float accuracy;
	float error;

public:
	Model();
	Model(int _nInput, int _nOutput);
	virtual ~Model();
	void setLoss(Loss _loss);
	void setOutputFunction(Activation _activation);
	void init();
	void fit(int _total, int _epoch, int _batch = 1);
	void evaluate(int _batch, bool show = false);
	void feed_forward(int idx);
	void back_propagation(int idx);
	void update();
	void addLayer(Layer* _layer);
	void addLayers(std::vector<Layer*>& _layers);
	void setInput(std::vector<Vector<float>>& _input_set);
	void setTarget(std::vector<Vector<float>>& _target_set);
	void setOptimizer(Optimizer *_optimizer);
	int getOutput();
	float getAccuracy();
	float getError();
	int getDepth();
	bool check_success();
	void setLearningRate(float lr);



};

#endif
