#ifndef VANILLA_DNN_DENSE_LAYER_HPP
#define VANILLA_DNN_DENSE_LAYER_HPP

#include <vector>
#include <functional>
#include <memory>
#include <VanillaDNN/Layers/Layer.hpp>
#include <VanillaDNN/Math/Matrix/Matrix.hpp>
#include <VanillaDNN/Math/Vector/Vector.hpp>
#include <VanillaDNN/Functions/Functions.hpp>



class DenseLayer : public Layer{
public:
	DenseLayer();
	DenseLayer(int dim);
	DenseLayer(int dim, Activation _activation);

	Activation activation, activation_diff;


	virtual ~DenseLayer();
	virtual void feed_forward(int idx);
	virtual void back_propagation(int idx);
	virtual void update();
	virtual void init(int batch_size, Optimizer *_optimizer);
	
	int dim;
	using Activation = std::function<Vector<float>(Vector<float>&)>;
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



};

#endif
