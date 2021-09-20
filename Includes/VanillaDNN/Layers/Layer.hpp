#ifndef VANILLA_DNN_LAYER_HPP
#define VANILLA_DNN_LAYER_HPP

#include <vector>
#include <functional>
#include <memory>
#include <VanillaDNN/Math/Matrix/Matrix.hpp>
#include <VanillaDNN/Math/Vector/Vector.hpp>
#include <VanillaDNN/Functions/Functions.hpp>



class Layer {
protected:
	using Activation = std::function<Vector<float>(Vector<float>&)>;
	int nNueron;

public:
	Layer();
	Layer(int _nNueron);
	Layer(int _nNueron, Activation _activation);

	Activation activation, activation_diff;
	Layer *preLayer = nullptr, *postLayer=nullptr;
	Matrix<float> weight;
	Vector<float> bias;
	
	std::vector<Vector<float>> inputNeuron; // before activate : preLayer->weight * neuronOutput
	std::vector<Vector<float>> outputNeuron; // after actionte : actinvation(neuronInput)

	std::vector<Matrix<float>> dE_dw;
	std::vector<Vector<float>> dE_db;

	std::vector<Vector<float>> dE_do;
	std::vector<Vector<float>> dE_dz;

	std::vector<Vector<float>> do_dz;
	
	std::vector<Vector<float>> dz_db;
	std::vector<Vector<float>> dz_dw;


	virtual ~Layer();
	virtual void feed_forward(int idx) = 0;
	virtual void back_propagation(int idx) = 0;
	virtual void update(const Matrix<float>& dw,const Vector<float>& db) = 0;
	
	void connect(Layer * layer);
	void setActivation(Activation _activation);
	int getNueronCnt();



};

#endif
