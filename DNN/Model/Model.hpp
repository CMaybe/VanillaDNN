#ifndef VANILLA_DNN_MODEL_HPP
#define VANILLA_DNN_MODEL_HPP


#include<any>
#include<functional>
#include<vector>
#include<iostream>
#include<random>
#include"Math/Matrix/Matrix.hpp"
#include"Math/Vector/Vector.hpp"
#include"DNN/Layer/Layer.hpp"
#include "MNIST/MNIST.hpp"


class Model {
private:
	using Activation = std::function<Vector<float>(Vector<float>)>;
	using  Loss = std::function<float(Vector<float>, Vector<float>)>;
	using  Loss_diff = std::function<Vector<float>(Vector<float>, Vector<float>)>;
	Loss loss;
	Loss_diff loss_diff;
	std::vector<Layer*> Layers; // exclude input&output Layer;

	Vector<float> input;
	Vector<float> target;
	Vector<float> output;
	std::vector<Vector<float>> input_batch;

	std::vector<Vector<float>> input_set;
	std::vector<Vector<float>> target_set;

	Layer* inputLayer = nullptr;
	Layer* outputLayer = nullptr;
	int depth;
	int nInput, nOutput;
	int batch;
	int epoch;
	int nEval;
	int total;
	float accuracy;
	float error;
	float learning_rate;

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
	void addLayer(Layer* _layer);
	void addLayers(std::vector<Layer*>& _layers);
	void setInput(std::vector<Vector<float>>& _input_set);
	void setTarget(std::vector<Vector<float>>& _target_set);
	int getOutput();
	float getAccuracy();
	float getError();
	bool check_success();
	void setLearningLate(float lr);



};

#endif
