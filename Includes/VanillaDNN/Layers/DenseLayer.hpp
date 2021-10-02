#ifndef VANILLA_DNN_DENSE_LAYER_HPP
#define VANILLA_DNN_DENSE_LAYER_HPP

#include <vector>
#include <functional>
#include <memory>
#include <VanillaDNN/Layers/Layer.hpp>
#include <VanillaDNN/Math/Matrix/Matrix.hpp>
#include <VanillaDNN/Math/Vector/Vector.hpp>
#include <VanillaDNN/Functions/Functions.hpp>
#include <VanillaDNN/Functions/Optimizer.hpp>




class DenseLayer : public Layer{
	private:
	using Activation = std::function<Vector<float>(Vector<float>&)>;
	Activation activation, activation_diff;
	Optimizer *optimizer;
	
	int dim;
	Matrix<float> weight;
	Vector<float> bias;
	
	std::vector<Matrix<float>> batch_weight;
	std::vector<Vector<float>> batch_bias;
	
	
	std::vector<Vector<float>> input; 
	std::vector<Vector<float>> output;

	std::vector<Matrix<float>> dE_dw;
	std::vector<Vector<float>> dE_db;

	std::vector<Vector<float>> dE_do;
	std::vector<Vector<float>> dE_dz;

	std::vector<Vector<float>> do_dz;
	
	std::vector<Vector<float>> dz_db;
	std::vector<Vector<float>> dz_dw;
	
	DenseLayer *preLayer = nullptr, *postLayer=nullptr;
	
public:
	DenseLayer();
	DenseLayer(const int& dim);
	DenseLayer(const int& dim, Activation _activation);

	virtual ~DenseLayer();

	virtual void feed_forward(const int& idx);
	virtual void back_propagation(const int& idx);
	virtual void predict();
	virtual void update();
	virtual void init(int batch_size, Optimizer *_optimizer);
	virtual void setInput(const Vector<float>& _input,const int& idx);
	virtual void setError(const Vector<float>& error,const int& idx);
	virtual void connect(Layer * layer);
	virtual void setOptimizer(Optimizer *_optimizer);
	
	virtual Vector<float> getOutput(const int& idx);
	virtual Layer* getPostLayer();
	virtual Layer* getPreLayer();
	
	void setActivation(Activation _activation);



};

#endif
